"""
sandbox_env.py

Provides a high-level API for managing sandboxed execution using OverlayFS.
Supports initialization, cleanup, and toggling between dry-run and active modes.
"""

import os
import subprocess
from app.engine.sandbox.overlayfs_utils import mount_overlay, unmount_overlay
from app.engine.sandbox.dry_run_manager import DryRunManager

SANDBOX_ROOT = "/tmp/code_sandbox"
MERGED_DIR = os.path.join(SANDBOX_ROOT, "merged")
UPPER_DIR = os.path.join(SANDBOX_ROOT, "upper")
WORK_DIR = os.path.join(SANDBOX_ROOT, "work")
LOWER_DIR = "/"  # Root FS as lowerdir in dry-run

def setup_sandbox(dry_run: bool = True):
    os.makedirs(MERGED_DIR, exist_ok=True)
    os.makedirs(UPPER_DIR, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)

    if dry_run:
        mount_overlay(lower=LOWER_DIR, upper=UPPER_DIR, work=WORK_DIR, merged=MERGED_DIR)
        DryRunManager.activate()
    else:
        DryRunManager.deactivate()

def teardown_sandbox():
    if os.path.ismount(MERGED_DIR):
        unmount_overlay(MERGED_DIR)

    for path in [MERGED_DIR, UPPER_DIR, WORK_DIR]:
        subprocess.run(["rm", "-rf", path], check=False)
