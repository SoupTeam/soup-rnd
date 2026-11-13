import json, sys, tarfile
from pathlib import Path

D = Path(r"C:\Users\sydyk\Downloads\moonlight-quality-evidence-20261111-whole3")
out = {}
for arm in ("bf16_l2l", "oracle_block64_e4m3"):
    for L in (1024, 4096):
        want = f"train-{arm}-L{L}/steps.jsonl"
        found = None
        for name in (f"moonlight-resident-20261111t201548z-whole3-train-{arm}-L{L}.tar.gz",
                     "moonlight-resident-20261111t201548z-whole3-final-remaining-evidence.tar.gz"):
            with tarfile.open(D / name, "r:gz") as tf:
                for member in tf:  # streaming: stop at the file we need
                    if member.name.endswith(want):
                        first = tf.extractfile(member).readline()
                        found = json.loads(first)
                        break
            if found:
                break
        s = found["activation_storage"]
        comp = s["components"]
        o = found.get("oracle_before_update") or {}
        out[f"{arm}|{L}"] = {
            "store_host_bytes": s["host_tensor_storage_bytes"],
            "store_device_bytes": s["device_tensor_storage_bytes"],
            "payload": comp["payload_including_padding"]["host_bytes"],
            "scales": comp["scales"]["host_bytes"],
            "calibration": comp["calibration_vectors"]["host_bytes"],
            "oracle_host_reservation": o.get("host_reservation_bound_bytes"),
            "oracle_owned_host": o.get("owned_host_storage_bytes"),
        }
        print(arm, L, out[f"{arm}|{L}"], flush=True)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding="utf-8")
