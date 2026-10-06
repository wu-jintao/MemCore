#!/usr/bin/env python3
"""Read-only inventory. Does not print credentials or private request content."""
import json
import os
import platform
import shutil
import subprocess


def command(args):
    process = subprocess.run(args, text=True, capture_output=True, check=False)
    return {"status": process.returncode, "output": process.stdout.strip(),
            "error": process.stderr.strip()}


print(json.dumps({
    "platform": platform.platform(),
    "python": platform.python_version(),
    "cpu_count": os.cpu_count(),
    "memory": command(["free", "-b"]),
    "block_devices": command(["lsblk", "-J", "-o", "NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS,UUID"]),
    "root_disk": dict(zip(("total", "used", "free"), shutil.disk_usage("/"))),
    "listeners": command(["ss", "-ltn"]),
    "data_mount": command(["findmnt", "--target", "/srv/aml-data"]),
}, ensure_ascii=False, indent=2))
