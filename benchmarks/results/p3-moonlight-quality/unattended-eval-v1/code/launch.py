"""One-shot rental + launch for the unattended Moonlight evaluation (user-approved cap $30).

  python launch.py plan   # read-only: show the offer/volume it would rent and the totals
  python launch.py go     # rent, upload, start the driver, register the watchdog task

Refuses to run twice (state.json exists). state.json is written after EVERY paid step, so
the watchdog can delete the resources even if this script dies half way. If the recorded
offer is gone it stops; it never rents a different machine.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE = HERE / "state.json"
REMOTE = HERE / "remote"
VAST = r"C:\Users\sydyk\AppData\Local\Programs\Python\Python314\Scripts\vastai.exe"
SSH = r"C:\Windows\System32\OpenSSH\ssh.exe"
SCP = r"C:\Windows\System32\OpenSSH\scp.exe"
KEY = r"C:\Users\sydyk\.ssh\e1_vast_ed25519"
PUBKEY = Path(KEY + ".pub")
PYTHON = str(Path(sys.executable).with_name("pythonw.exe"))  # no console window per tick
IMAGE = "ubuntu@sha256:78d4744ead1f5b19b04d0df558ab182a5300903900755e515b2ca4d10eb2044d"
MACHINE_ID = 140544
OFFER_ID = 59917583      # RTX 4090, same machine as the qualified first host
VOLUME_OFFER_ID = 59917588
VOLUME_GB = 220
ROOT_GB = 32
HOURLY_USD = 0.62        # 0.5319 gpu/root (earlier quote) + 0.0815 volume, rounded up
PRIOR_SPEND_USD = 8.30   # posted $7.904 for all 20 earlier resources (incl. the deleted attempt 2,
#                          $0.709) at 2026-11-12 09:15Z, plus a margin for charges still settling
TRAFFIC_USD = 0.60       # ~36 GB down at $0.0067/GB plus uploads/downloads, rounded up
COMPUTE_STOP_USD = 28.5  # watchdog asks the run to stop: leaves ~$1.5 of the $30 cap for the
#                          current piece (<= ~25 min), packaging and the download
HARD_CAP_USD = 29.5      # watchdog deletes even without an archive
STOP_UTC = "2026-11-13T21:00:00+00:00"
HARD_CLOSE_UTC = "2026-11-13T21:45:00+00:00"
OLD_IDS = {  # every resource of this study so far, including the deleted attempt 2
    "instances": [60241118, 60244148, 60246341, 60256341, 60257559, 60258336, 60259585,
                  60260641, 60261779, 60329118],
    "volumes": [60241117, 60244147, 60246278, 60256277, 60257498, 60258329, 60259577,
                60260624, 60261770, 60329113],
}
REMOTE_ROOT = "/data/p3-moonlight-quality"


def now() -> datetime:
    return datetime.now(timezone.utc)


def vast(*args: str, stdin: str | None = None, timeout: int = 180):
    done = subprocess.run([VAST, *args, "--raw"], capture_output=True, text=True,
                          input=stdin, timeout=timeout, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"vastai {' '.join(args[:2])} failed: {done.stderr or done.stdout}")
    value = json.loads(done.stdout)
    # The CLI exits 0 and prints {"error": true, "status_code": ..., "msg": ...} on rejection.
    if isinstance(value, dict) and (value.get("error") or value.get("success") is False):
        raise RuntimeError(f"vastai {' '.join(args[:2])} rejected: {value}")
    return value


def save(state: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    tmp.replace(STATE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_offers() -> tuple[dict, dict]:
    offers = vast("search", "offers", f"machine_id={MACHINE_ID} rentable=true gpu_frac=1")
    offer = next((o for o in offers if o["id"] == OFFER_ID), None)
    if offer is None or offer.get("gpu_name") != "RTX 4090" or offer.get("num_gpus") != 1:
        raise SystemExit("OFFER_GONE: the recorded RTX 4090 offer is unavailable. Nothing was "
                         "rented. Do not rent another machine without asking the user.")
    volumes = vast("search", "volumes", f"machine_id={MACHINE_ID}")
    volume = next((v for v in volumes if v["id"] == VOLUME_OFFER_ID), None)
    if volume is None:
        raise SystemExit("VOLUME_OFFER_GONE: no matching volume offer. Nothing was rented.")
    return offer, volume


def plan() -> None:
    offer, volume = find_offers()
    hourly = offer["dph_total"] + volume["storage_total_cost"] * VOLUME_GB
    print(json.dumps({
        "offer": {k: offer.get(k) for k in ("id", "machine_id", "gpu_name", "dph_total",
                                            "reliability", "driver_version", "geolocation")},
        "volume_offer": {k: volume.get(k) for k in ("id", "disk_space", "storage_total_cost",
                                                    "disk_name")},
        "quoted_hourly_usd": round(hourly, 4), "budget_hourly_usd": HOURLY_USD,
        "hours_to_stop_usd": round((COMPUTE_STOP_USD - PRIOR_SPEND_USD - TRAFFIC_USD) / HOURLY_USD, 1),
        "now_utc": now().isoformat(), "stop_utc": STOP_UTC}, indent=2))


def ssh_cmd(state: dict, command: str, timeout: int = 120) -> tuple[int, str]:
    base = [SSH, "-i", KEY, "-p", str(state["ssh_port"]), "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=20", "-o", "IdentitiesOnly=yes",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={(HERE / 'known_hosts').as_posix()}",
            f"root@{state['ssh_host']}"]
    try:
        done = subprocess.run([*base, command], capture_output=True, text=True, timeout=timeout,
                              check=False)
    except subprocess.TimeoutExpired:
        return 255, "timeout"
    return done.returncode, done.stdout + done.stderr


def scp_up(state: dict, local: Path, remote: str, timeout: int = 1800) -> None:
    command = [SCP, "-i", KEY, "-P", str(state["ssh_port"]), "-o", "BatchMode=yes",
               "-o", "ConnectTimeout=20", "-o", "IdentitiesOnly=yes",
               "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6",
               "-o", "StrictHostKeyChecking=accept-new",
               "-o", f"UserKnownHostsFile={(HERE / 'known_hosts').as_posix()}", str(local),
               f"root@{state['ssh_host']}:{remote}"]
    done = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"scp failed: {done.stderr}")


def build_bundle() -> Path:
    bundle = HERE / "inputs-bundle.tar.gz"
    if bundle.exists():
        bundle.unlink()
    with tarfile.open(bundle, "w:gz") as tar:
        for path in sorted(REMOTE.iterdir()):
            if path.name in ("inputs", "__pycache__"):
                continue
            tar.add(str(path), arcname=path.name)
        tar.add(str(REMOTE / "inputs"), arcname="inputs")
    return bundle


def register_watchdog(state: dict) -> None:
    """Register the 15-minute watchdog BEFORE any paid resource exists, so every later
    failure (creation, reachability, host admission, upload) is closed by the watchdog."""
    task = subprocess.run(
        ["schtasks", "/Create", "/F", "/TN", "MoonlightWatchdog", "/SC", "MINUTE", "/MO", "15",
         "/RL", "LIMITED", "/TR",
         f'"{PYTHON}" -X utf8 "{HERE / "moonlight_watchdog.py"}" tick'],
        capture_output=True, text=True, check=False)
    if task.returncode == 0:  # 3 h cap per tick, wake-to-run, no stop on battery
        subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "$t=Get-ScheduledTask MoonlightWatchdog;$s=$t.Settings;$s.WakeToRun=$true;"
             "$s.StartWhenAvailable=$true;$s.DisallowStartIfOnBatteries=$false;"
             "$s.StopIfGoingOnBatteries=$false;$s.ExecutionTimeLimit='PT3H';"
             "Set-ScheduledTask -InputObject $t | Out-Null"],
            capture_output=True, text=True, check=False)
    print("schtasks:", task.returncode, (task.stdout + task.stderr).strip())
    state["watchdog_task_registered"] = task.returncode == 0
    save(state)
    if task.returncode != 0:
        STATE.unlink(missing_ok=True)
        raise SystemExit("could not register the watchdog task; nothing was rented")


def go() -> None:
    if STATE.exists():
        raise SystemExit("state.json exists: a rental was already started. Not renting again.")
    offer, volume = find_offers()
    all_ids = OLD_IDS["instances"] + OLD_IDS["volumes"]
    state = {
        "created_utc": now().isoformat(), "ssh_host": None, "ssh_port": None,
        "instance_id": None, "volume_id": None, "all_owned_ids": list(all_ids),
        "hourly_usd": HOURLY_USD, "prior_spend_usd": PRIOR_SPEND_USD, "traffic_usd": TRAFFIC_USD,
        "compute_stop_usd": COMPUTE_STOP_USD, "hard_cap_usd": HARD_CAP_USD,
        "stop_utc": STOP_UTC, "hard_close_utc": HARD_CLOSE_UTC, "stage": "created",
    }
    save(state)
    register_watchdog(state)
    try:
        created = vast("create", "volume", str(VOLUME_OFFER_ID), "-s", str(VOLUME_GB), "-n",
                       "moonlight_h2_eval")
    except (RuntimeError, ValueError) as error:
        STATE.unlink(missing_ok=True)
        subprocess.run(["schtasks", "/Delete", "/TN", "MoonlightWatchdog", "/F"],
                       capture_output=True, check=False)
        raise SystemExit(f"volume creation failed: {error}. Nothing was rented.") from error
    state["volume_id"] = created["volume_id"]
    state["all_owned_ids"].append(created["volume_id"])
    state["created_utc"] = now().isoformat()
    save(state)
    print("volume", state["volume_id"])
    pubkey = PUBKEY.read_text(encoding="utf-8").strip()
    onstart = ("install -d -m 700 /root/.ssh; printf '%s\\n' '" + pubkey +
               "' >> /root/.ssh/authorized_keys; chmod 600 /root/.ssh/authorized_keys")
    try:
        instance = vast("create", "instance", str(OFFER_ID), "--image", IMAGE, "--disk",
                        str(ROOT_GB), "--ssh", "--direct", "--link-volume",
                        str(state["volume_id"]), "--mount-path", "/data", "--label",
                        "moonlight-h2-eval", "--onstart-cmd", onstart, "--cancel-unavail")
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"instance creation failed: {error}. The volume {state['volume_id']} "
                         f"exists; the watchdog deletes it within 90 minutes, or run: "
                         f"python moonlight_watchdog.py close --force") from error
    state["instance_id"] = int(instance["new_contract"])
    state["all_owned_ids"].append(state["instance_id"])
    save(state)
    print("instance", state["instance_id"])

    # `vastai attach ssh` prints a Python dict, not JSON, so read it as text. The key was also
    # injected through --onstart-cmd, so a failed attach is not fatal.
    attach = subprocess.run([VAST, "attach", "ssh", str(state["instance_id"]), pubkey],
                            capture_output=True, text=True, check=False, timeout=120)
    print("attach ssh:", (attach.stdout + attach.stderr).strip()[:200])
    for attempt in range(60):
        try:
            rows = [r for r in vast("show", "instances") if r["id"] == state["instance_id"]]
        except (RuntimeError, ValueError) as error:
            print("poll:", error)
            time.sleep(20)
            continue
        row = rows[0] if rows else {}
        # prefer the direct endpoint (public IP + mapped 22/tcp) over Vast's proxy
        mapping = (row.get("ports") or {}).get("22/tcp") or []
        if mapping and row.get("public_ipaddr"):
            host, port = row["public_ipaddr"], mapping[0]["HostPort"]
        else:
            host, port = row.get("ssh_host"), row.get("ssh_port")
        if row.get("actual_status") == "running" and host and port:
            state.update(ssh_host=host.strip(), ssh_port=int(port), machine_id=row.get("machine_id"))
            save(state)
            code, out = ssh_cmd(state, "echo SSH_OK")
            if code == 0 and "SSH_OK" in out:
                break
        time.sleep(20)
    else:
        raise SystemExit("instance never became reachable; run: python moonlight_watchdog.py close")
    if state.get("machine_id") != MACHINE_ID:
        raise SystemExit(f"WRONG_MACHINE {state.get('machine_id')}; run: "
                         "python moonlight_watchdog.py close")
    code, out = ssh_cmd(state, "findmnt -no SOURCE,FSTYPE /data; nvidia-smi --query-gpu=name,"
                               "driver_version --format=csv,noheader; nproc; free -g | head -2")
    print(out)
    if "xfs" not in out or "RTX 4090" not in out:
        raise SystemExit("HOST_NOT_ADMITTED: /data is not xfs or GPU is not an RTX 4090; run: "
                         "python moonlight_watchdog.py close")
    bundle = build_bundle()
    digest = sha256(bundle)
    code, out = ssh_cmd(state, f"mkdir -p {REMOTE_ROOT}/autorun")
    scp_up(state, bundle, f"{REMOTE_ROOT}/autorun-inputs.tar.gz")
    code, out = ssh_cmd(state, f"cd {REMOTE_ROOT} && echo '{digest}  autorun-inputs.tar.gz' | "
                               f"sha256sum -c - && tar -xzf autorun-inputs.tar.gz -C "
                               f"{REMOTE_ROOT}/autorun && ls {REMOTE_ROOT}/autorun")
    print(out)
    if code != 0:
        raise SystemExit("upload verification failed; run: python moonlight_watchdog.py close")
    code, out = ssh_cmd(state, "command -v python3 >/dev/null || (apt-get update -qq && "
                               "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3 "
                               "ca-certificates); python3 --version", timeout=900)
    print(out)
    if code != 0:
        raise SystemExit("python3 is unavailable on the host; run: "
                         "python moonlight_watchdog.py close")
    code, out = ssh_cmd(state, f"bash {REMOTE_ROOT}/autorun/launch_driver.sh", timeout=60)
    started = False
    for _ in range(24):
        time.sleep(10)
        code, out = ssh_cmd(state, f"cat {REMOTE_ROOT}/autorun/status.json 2>/dev/null")
        if '"state": "RUNNING"' in out:
            started = True
            break
    print("driver status:", out.strip()[:300])
    if not started:
        raise SystemExit("driver did not report RUNNING within 4 minutes; the watchdog will "
                         "close the host, or run: python moonlight_watchdog.py close --force")
    state["stage"] = "driver_started"
    state["driver_started_utc"] = now().isoformat()
    save(state)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if command == "plan":
        plan()
    elif command == "go":
        go()
    else:
        print(__doc__)
        sys.exit(2)
