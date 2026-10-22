"""Independently validate a downloaded development archive without CUDA."""

import argparse
import hashlib
import json
import zipfile
from xml.etree import ElementTree

parser = argparse.ArgumentParser()
parser.add_argument("archive")
parser.add_argument("expected_sha256")
parser.add_argument("output")
args = parser.parse_args()
with open(args.archive, "rb") as handle:
    assert hashlib.sha256(handle.read()).hexdigest() == args.expected_sha256
with zipfile.ZipFile(args.archive) as zf:
    assert zf.testzip() is None
    assert len(zf.namelist()) == len(set(zf.namelist()))
    manifest = json.loads(zf.read("ARCHIVE-SHA256.json"))
    assert set(zf.namelist()) == set(manifest) | {"ARCHIVE-SHA256.json"}
    for name, expected in manifest.items():
        assert hashlib.sha256(zf.read(name)).hexdigest() == expected, name
    receipt = json.loads(zf.read("cuda-development-01/receipt.json"))
    for name, expected in receipt["sources"].items():
        assert hashlib.sha256(zf.read(name)).hexdigest() == expected, name
    cases = list(ElementTree.fromstring(zf.read("cuda-development-01/junit.xml")).iter(
        "testcase"
    ))
    collected = {line.strip() for line in zf.read("cuda-development-01/collection.log").decode(
        ).splitlines() if line.startswith("tests/")}
    executed = {case.get("classname").replace(".", "/") + ".py::" + case.get("name")
                for case in cases}
    assert collected == executed
    assert len(cases) == len(executed)
    assert all(case.find(tag) is None for case in cases for tag in ("failure", "error", "skipped"))
    assert len(receipt["cases"]) == len(cases)
    assert receipt["returncode"] == 0
    memory = {prop.get("name"): prop.get("value") for case in cases if
              "reduces_measured_allocated_peak" in case.get("name")
              for prop in case.findall("./properties/property")}
    result = {"kind": "offline development evidence verification, no CUDA replay",
              "archive_sha256": args.expected_sha256, "payload_hashes": len(manifest),
              "source_hashes": len(receipt["sources"]), "passed_cases": len(cases),
              "collected_equals_executed": True, "zip_crc": "passed", "activation_peak": memory,
              "preflight": json.loads(zf.read("preflight.json"))}
with open(args.output, "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2)
    handle.write("\n")
print(json.dumps({k: v for k, v in result.items() if k != "preflight"}, indent=2))
