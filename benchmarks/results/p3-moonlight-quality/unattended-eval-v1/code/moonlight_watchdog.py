"""Local watchdog for the unattended Moonlight rental (user-authorized, 2026-11-12).

Commands:
  status            print the remote step, progress and cost estimate (read-only)
  tick              one supervision pass (Windows Task Scheduler runs this every 15 minutes)
  download          download and verify the final archive now; never deletes anything
  close [--force]   delete the owned instance and volume. Refuses while the run is still
                    RUNNING, or while a finished archive could not be downloaded, unless --force.

Only the instance and volume recorded in state.json are ever touched.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
STATE = HERE / "state.json"
LOG = HERE / "watchdog.log"
RESULTS = Path(r"C:\Users\sydyk\Downloads\moonlight-quality-results-h2")
VAST = r"C:\Users\sydyk\AppData\Local\Programs\Python\Python314\Scripts\vastai.exe"
SSH = r"C:\Windows\System32\OpenSSH\ssh.exe"
SCP = r"C:\Windows\System32\OpenSSH\scp.exe"
KEY = r"C:\Users\sydyk\.ssh\e1_vast_ed25519"
TASK_NAME = "MoonlightWatchdog"
ROOT = "/data/p3-moonlight-quality"
REMOTE_AUTO = f"{ROOT}/autorun"
LOCK = f"{REMOTE_AUTO}/driver.lock"
LAUNCH = f"{REMOTE_AUTO}/launch_driver.sh"
RUN_ID = "moonlight-resident-20261111t201548z-whole3"
BILLING_START = "2026-11-11"
ENDED = ("DONE", "FAILED", "STOPPED")
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def now() -> datetime:
    return datetime.now(timezone.utc)


def log(message: str) -> None:
    line = f"{now().isoformat()} {message}"
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    try:
        print(line)
    except Exception:  # noqa: BLE001 - logging must never block a delete
        pass


def run(command: list[str], timeout: int, stdin: str | None = None):
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, input=stdin,
                          creationflags=NO_WINDOW, check=False)


def load() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8"))


def save(state: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    tmp.replace(STATE)


def ssh_options(state: dict) -> list[str]:
    return ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "IdentitiesOnly=yes",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={(HERE / 'known_hosts').as_posix()}"]


def remote(state: dict, command: str, timeout: int = 120) -> tuple[int, str]:
    if not state.get("ssh_host"):
        return 255, "no ssh_host"
    base = [SSH, "-i", KEY, "-p", str(state["ssh_port"]), *ssh_options(state),
            f"root@{state['ssh_host']}"]
    try:
        done = run([*base, command], timeout)
    except subprocess.TimeoutExpired:
        return 255, "timeout"
    return done.returncode, done.stdout


def remote_status(state: dict) -> dict | None:
    """status.json plus a kernel-owned liveness probe: the driver holds a flock on driver.lock
    for its whole life, so `flock -n` failing means it is alive (no pgrep self-match, no procps)."""
    code, out = remote(
        state, f"cat {REMOTE_AUTO}/status.json 2>/dev/null; echo; "
               f"if flock -n {LOCK} true 2>/dev/null; then echo DRIVER_DEAD; "
               f"else echo DRIVER_ALIVE; fi")
    if code != 0:
        return None
    start, end = out.find("{"), out.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(out[start:end + 1])
    except ValueError:
        return None
    value["driver_alive"] = "DRIVER_ALIVE" in out
    return value


def vast_json(*args: str):
    done = run([VAST, *args, "--raw"], 120)
    if done.returncode != 0 or not done.stdout.strip():
        raise RuntimeError((done.stderr or done.stdout or "empty response").strip())
    value = json.loads(done.stdout)
    if isinstance(value, dict) and value.get("error"):
        raise RuntimeError(f"vastai {' '.join(args[:2])} rejected: {value}")
    return value


def vast_text(*args: str, stdin: str | None = None) -> str:
    """The vastai CLI exits 0 even on failure; the text and a later absence check are the truth."""
    done = run([VAST, *args], 180, stdin=stdin)
    return (done.stdout + done.stderr).strip()


def posted_usd(ids: list) -> float | None:
    """Posted charges for every resource of this study (all hosts), all invoice pages."""
    wanted = [str(item) for item in ids]
    total, token = 0.0, None
    end = (now() + timedelta(days=1)).date().isoformat()
    try:
        while True:
            args = ["show", "invoices-v1", "--charges", "--start-date", BILLING_START,
                    "--end-date", end, "--limit", "100", "--latest-first"]
            if token:
                args += ["--next-token", token]
            page = vast_json(*args)
            for row in page.get("results", []):
                text = row.get("description", "")
                if any(re.search(rf"(?<!\d){item}(?!\d)", text) for item in wanted):
                    total += float(row["amount"])
            token = page.get("next_token")
            if not token:
                return total
    except Exception as error:  # noqa: BLE001 - billing visibility loss is logged, not fatal
        log(f"billing read failed: {error}")
        return None


def cost_estimate(state: dict) -> dict:
    hours = (now() - datetime.fromisoformat(state["created_utc"])).total_seconds() / 3600
    accrued = state["prior_spend_usd"] + hours * state["hourly_usd"] + state["traffic_usd"]
    posted = posted_usd(state["all_owned_ids"])
    return {"hours_rented": round(hours, 2), "accrued_estimate_usd": round(accrued, 3),
            "posted_usd": posted, "estimate_usd": round(max(accrued, posted or 0.0), 3)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(state: dict, status: dict) -> bool:
    archive, digest = status.get("archive"), status.get("archive_sha256")
    if not archive or not digest:
        return False
    RESULTS.mkdir(parents=True, exist_ok=True)
    target = RESULTS / Path(archive).name
    if not (target.exists() and sha256(target) == digest):
        part = target.with_name(target.name + ".part")
        part.unlink(missing_ok=True)
        command = [SCP, "-i", KEY, "-P", str(state["ssh_port"]), *ssh_options(state),
                   f"root@{state['ssh_host']}:{archive}", str(part)]
        try:
            done = run(command, 7200)
        except subprocess.TimeoutExpired:
            log("download timed out")
            return False
        if done.returncode != 0:
            log(f"download failed: {done.stderr.strip()[-300:]}")
            return False
        if sha256(part) != digest:
            log("download hash mismatch; discarded")
            part.unlink(missing_ok=True)
            return False
        part.replace(target)
    (RESULTS / "remote-status.json").write_text(json.dumps(status, indent=2) + "\n",
                                                 encoding="utf-8")
    log(f"downloaded and verified {target} sha256={digest}")
    state["downloaded"] = {"path": str(target), "sha256": digest, "utc": now().isoformat()}
    save(state)
    return True


def rescue_and_fetch(state: dict) -> bool:
    """No verified archive exists: tar whatever evidence is there, fetch it and the logs."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    code, out = remote(
        state, f"cd {ROOT} && mkdir -p exports && tar --ignore-failed-read -czf "
               f"exports/rescue.tar.gz --exclude='*/train-*' --exclude=autorun/inputs autorun "
               f"evidence/{RUN_ID} && sha256sum exports/rescue.tar.gz", timeout=3000)
    rescued = False
    match = re.search(r"([0-9a-f]{64})\s+exports/rescue\.tar\.gz", out) if code == 0 else None
    if match:
        rescued = download(state, {"archive": f"{ROOT}/exports/rescue.tar.gz",
                                   "archive_sha256": match.group(1), "state": "RESCUE"})
    target = RESULTS / "no-archive"
    target.mkdir(exist_ok=True)
    for name in ("status.json", "driver.log", "driver-steps.log", "logs"):
        command = [SCP, "-r", "-i", KEY, "-P", str(state["ssh_port"]), *ssh_options(state),
                   f"root@{state['ssh_host']}:{REMOTE_AUTO}/{name}", str(target)]
        try:
            run(command, 900)
        except subprocess.TimeoutExpired:
            log(f"timed out fetching {name}")
    log(f"rescue attempt finished; archive_rescued={rescued}; status and logs in {target}")
    return rescued


def owned_resources() -> tuple[dict[int, dict], dict[int, dict]]:
    instances = {int(row["id"]): row for row in vast_json("show", "instances")}
    raw = vast_json("show", "volumes")
    rows = raw if isinstance(raw, list) else raw.get("volumes", [])
    return instances, {int(row["id"]): row for row in rows}


def destroy(state: dict, reason: str) -> bool:
    """Delete the recorded instance, then the volume, until both are verifiably absent.
    `closed` is only latched on a positive absence reading; otherwise the next tick retries."""
    iid = int(state["instance_id"]) if state.get("instance_id") else None
    vid = int(state["volume_id"]) if state.get("volume_id") else None
    log(f"closing owned resources instance={iid} volume={vid}: {reason}")
    receipts = state.setdefault("close_receipts", [])
    stop_at = time.monotonic() + 900
    absent = False
    confirmed_once = False
    while time.monotonic() < stop_at:
        try:
            instances, volumes = owned_resources()
        except Exception as error:  # noqa: BLE001
            log(f"absence check failed: {error}")
            time.sleep(30)
            continue
        if iid and iid in instances:
            wanted = state.get("machine_id")
            if wanted and int(instances[iid].get("machine_id", -1)) != int(wanted):
                log(f"REFUSING to delete {iid}: machine_id differs from state")
                return False
            receipts.append({"kind": "instance", "id": iid,
                             "out": vast_text("destroy", "instance", str(iid), stdin="y\n")[-300:]})
        elif vid and vid in volumes:
            receipts.append({"kind": "volume", "id": vid,
                             "out": vast_text("delete", "volume", str(vid))[-300:]})
        else:
            if not confirmed_once:  # one empty API answer must not latch `closed`
                confirmed_once = True
                time.sleep(20)
                continue
            absent = True
            break
        log(f"close step sent: {receipts[-1]}")
        time.sleep(30)
    state["close_attempts"] = state.get("close_attempts", 0) + 1
    if absent:
        state["closed"] = {"utc": now().isoformat(), "reason": reason, "verified_absent": True}
        save(state)
        log("closed; verified_absent=True")
        run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"], 60)
    else:
        save(state)
        log(f"NOT CLOSED after {state['close_attempts']} attempt(s); the next tick retries")
    return absent


def request_stop(state: dict) -> bool:
    code, out = remote(state, f"touch {REMOTE_AUTO}/STOP && test -e {REMOTE_AUTO}/STOP "
                              f"&& echo STOP_OK")
    return code == 0 and "STOP_OK" in out


def instance_row(state: dict) -> dict | None:
    try:
        return next((row for row in vast_json("show", "instances")
                     if int(row["id"]) == int(state["instance_id"])), None)
    except Exception as error:  # noqa: BLE001
        log(f"instance lookup failed: {error}")
        return None


def tick() -> None:
    state = load()
    if state.get("closed"):
        return
    if state.get("stage") != "driver_started":
        # launch.py is still starting the rental, or it died. If the driver is in fact running
        # (launch.py died after starting it), adopt the run instead of closing it.
        probe = remote_status(state) if state.get("ssh_host") else None
        if probe and probe.get("state") == "RUNNING" and probe.get("driver_alive"):
            state["stage"] = "driver_started"
            save(state)
            log("driver is running although launch.py did not finish; adopting the run")
        else:
            age = (now() - datetime.fromisoformat(state["created_utc"])).total_seconds()
            if age > 5400:
                destroy(state, f"launch never reached driver_started in 90 minutes "
                               f"(stage={state.get('stage')})")
            else:
                log(f"launch in progress (stage={state.get('stage')}, {int(age)} s)")
            return
    status = remote_status(state)
    cost = cost_estimate(state)
    state["last_tick"] = {"utc": now().isoformat(), "status": status, "cost": cost}
    save(state)
    log(f"tick state={status and status.get('state')} step={status and status.get('step')} "
        f"shard={status and status.get('shard_index')}/{status and status.get('shard_total')} "
        f"alive={status and status.get('driver_alive')} cost={cost}")
    if status is None:
        row = instance_row(state)
        log(f"remote unreachable; provider reports status={row and row.get('actual_status')}")
    deadline = datetime.fromisoformat(state["stop_utc"])
    hard = datetime.fromisoformat(state["hard_close_utc"])
    over_budget = cost["estimate_usd"] >= state["compute_stop_usd"]
    hard_hit = cost["estimate_usd"] >= state["hard_cap_usd"] or now() >= hard
    ended = bool(status) and status.get("state") in ENDED
    has_archive = ended and bool(status.get("archive_sha256"))

    if has_archive:
        # An early FAILED (before any scoring piece completed) is usually a defect in the run's
        # own checks, not a scientific result, and a repaired relaunch is far cheaper on the same
        # host (the 32 GB weights and the cache are already there). Keep the host for a bounded
        # repair window; the archive is downloaded first either way.
        early = status.get("state") == "FAILED" and not state.get("repair_window_used")
        if early and "completed_pieces" in status and status.get("completed_pieces", 0) == 0:
            began = state.setdefault("repair_window_started_utc", now().isoformat())
            held = (now() - datetime.fromisoformat(began)).total_seconds()
            if held < 5400 and not hard_hit:
                download(state, status)
                save(state)
                log(f"early failure ({status.get('detail')}); keeping the host for repair "
                    f"({int(held)} s of 5400 used)")
                return
            state["repair_window_used"] = True
            save(state)
        if download(state, status):
            destroy(state, f"run ended: {status.get('state')} {status.get('detail')}")
            return
        state["download_attempts"] = state.get("download_attempts", 0) + 1
        save(state)
        log(f"download failed ({state['download_attempts']}); keeping the host until a download "
            f"succeeds or the hard close")
        if not hard_hit:
            return
    if hard_hit:
        for _ in range(3):  # last chance before deleting
            if status:
                break
            time.sleep(60)
            status = remote_status(state)
        if status and status.get("archive_sha256") and status.get("state") in ENDED:
            download(state, status)
        elif not state.get("downloaded") and not state.get("rescue_attempted"):
            state["rescue_attempted"] = now().isoformat()
            save(state)
            rescue_and_fetch(state)
        destroy(state, f"hard cap/deadline reached: {cost}")
        return
    if ended:  # ended without a verified archive (packaging failed): one rescue, then close
        if not state.get("rescue_attempted"):
            state["rescue_attempted"] = now().isoformat()
            save(state)
            rescue_and_fetch(state)
        destroy(state, f"run ended without archive: {status.get('state')} {status.get('detail')}")
        return
    if status and status.get("state") == "RUNNING" and not status.get("driver_alive"):
        relaunches = state.get("relaunches", 0)
        if relaunches < 2:
            code, out = remote(state, f"bash {LAUNCH}", timeout=60)
            if code == 0:
                state["relaunches"] = relaunches + 1
                save(state)
                log(f"driver not running; relaunch {relaunches + 1}/2 sent")
            else:
                log(f"relaunch failed rc={code} {out[-200:]}")
        else:
            if not state.get("rescue_attempted"):
                state["rescue_attempted"] = now().isoformat()
                save(state)
                rescue_and_fetch(state)
            destroy(state, "driver died repeatedly; both relaunches used")
        return
    if over_budget or now() >= deadline:
        if not state.get("stop_requested_utc"):
            if request_stop(state):
                state["stop_requested_utc"] = now().isoformat()
                save(state)
                log(f"STOP delivered: over_budget={over_budget} past_deadline={now() >= deadline}")
            else:
                log("STOP not delivered (ssh failed); will retry next tick")


def print_status() -> None:
    state = load()
    closed = state.get("closed")
    status = None if closed or not state.get("ssh_host") else remote_status(state)
    cost = cost_estimate(state) if state.get("created_utc") else None
    print(json.dumps({"remote": status, "cost": cost, "downloaded": state.get("downloaded"),
                      "closed": closed, "stop_requested_utc": state.get("stop_requested_utc"),
                      "results_folder": str(RESULTS)}, indent=2))

def print_progress() -> None:
    """One readable line; says plainly whether the host was deleted or just not answering."""
    if not STATE.exists():
        print("Нет активной аренды (state.json отсутствует).")
        return
    state = load()
    closed = state.get("closed")
    cost = cost_estimate(state) if state.get("created_utc") else {}
    money = f"~${cost.get('estimate_usd', '?')} из $30"
    if closed:
        print(f"ЗАКРЫТО: машина и диск удалены ({closed.get('reason', '')[:90]}). "
              f"Расход {money}. Результаты: {RESULTS}")
        return
    status = remote_status(state) if state.get("ssh_host") else None
    if status is None:
        print(f"Машина существует, но не отвечает по SSH (не удалена). Расход {money}. "
              f"Сторож продолжит попытки каждые 15 минут; ничего не удалит до 21:45 UTC 13.11.")
        return
    step, total = status.get("step"), status.get("shard_total")
    part = f", кусок {status.get('shard_index')} из {total}" if total else ""
    done = f"; итоговые архивы: {status.get('archive')}" if status.get("archive") else ""
    print(f"{status.get('state')} · шаг {step}{part} · драйвер "
          f"{'жив' if status.get('driver_alive') else 'НЕ работает'} · {money}{done}")



def single_instance():
    import msvcrt
    handle = open(HERE / "watchdog.lock", "a+")  # noqa: SIM115 - held for the process lifetime
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        log("another watchdog command is running; exiting")
        sys.exit(0)
    return handle


def command_main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    force = "--force" in sys.argv
    if command == "status":
        print_status()
        return 0
    if command == "progress":
        print_progress()
        return 0
    _lock = single_instance()  # noqa: F841
    if command == "tick":
        tick()
        return 0
    state = load()
    status = None if state.get("closed") or not state.get("ssh_host") else remote_status(state)
    has_archive = bool(status) and status.get("state") in ENDED and status.get("archive_sha256")
    if command == "download":
        if not has_archive:
            print("Run has not ended with an archive yet; nothing to download.")
            return 1
        return 0 if download(state, status) else 1
    if command == "close":
        if status is None and state.get("ssh_host") and not state.get("closed") and not force:
            print("The host is unreachable, so the run's state is unknown. NOT deleting. "
                  "Check `status`, or use `close --force`.")
            return 1
        if status and status.get("state") == "RUNNING" and not force:
            print("The run is still RUNNING. `close --force` deletes it unfinished.")
            return 1
        if has_archive and not download(state, status) and not force:
            print("The archive exists but could not be downloaded. NOT deleting. "
                  "Re-run, or use `close --force`.")
            return 1
        return 0 if destroy(state, "manual close") else 1
    print(__doc__)
    return 2


def main() -> int:
    try:
        return command_main()
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 - Task Scheduler discards stderr; leave a trace
        log("UNHANDLED " + traceback.format_exc().replace("\n", " | "))
        return 1


if __name__ == "__main__":
    sys.exit(main())
