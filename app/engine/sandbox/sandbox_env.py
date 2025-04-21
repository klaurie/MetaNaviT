"""
sandbox_env.py

Provides a high-level API for managing sandboxed execution using OverlayFS.
Supports initialization, cleanup, and toggling between dry-run and active modes.
"""

import os
import subprocess
import logging
from app.engine.sandbox.overlayfs_utils import mount_overlay, unmount_overlay
from app.engine.sandbox.dry_run_manager import DryRunManager

logger = logging.getLogger(__name__)

SANDBOX_ROOT = "/tmp/code_sandbox"
MERGED_DIR = os.path.join(SANDBOX_ROOT, "merged")
UPPER_DIR = os.path.join(SANDBOX_ROOT, "upper")
WORK_DIR = os.path.join(SANDBOX_ROOT, "work")
LOWER_DIR = "/"  # Root FS as lowerdir in dry-run

def setup_sandbox(dry_run: bool = True):
    """
    Set up the sandbox environment using OverlayFS.

    Args:
        dry_run (bool): If True, activates dry-run mode; otherwise, sets up a writable sandbox.
    """
    print(f"[DEBUG] Creating sandbox directories at {SANDBOX_ROOT}")
    os.makedirs(MERGED_DIR, exist_ok=True)
    os.makedirs(UPPER_DIR, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)

    try:
        if dry_run:
            logger.info("Setting up sandbox in dry-run mode.")
            print(f"[DEBUG] Mounting sandbox in dry-run mode:")
            mount_overlay(lower=LOWER_DIR, upper=UPPER_DIR, work=WORK_DIR, merged=MERGED_DIR)
            DryRunManager.activate()
        else:
            logger.info("Setting up sandbox in writable mode.")
            print(f"[DEBUG] Skipping mount — writable mode not implemented yet.")
            DryRunManager.deactivate()
    except Exception as e:
        logger.error(f"Failed to set up sandbox: {e}")
        raise

def teardown_sandbox():
    """
    Tear down the sandbox environment by unmounting OverlayFS and cleaning up directories.
    """
    try:
        if os.path.ismount(MERGED_DIR):
            logger.info("Unmounting sandbox OverlayFS.")
            print(f"[DEBUG] Unmounting OverlayFS at {MERGED_DIR}")
            unmount_overlay(MERGED_DIR)

        for path in [MERGED_DIR, UPPER_DIR, WORK_DIR]:
            print(f"[DEBUG] Removing path: {path}")
            subprocess.run(["rm", "-rf", path], check=False)

        logger.info("Sandbox environment cleaned up successfully.")
        print(f"[DEBUG] Sandbox teardown complete.")
    except Exception as e:
        logger.error(f"Failed to tear down sandbox: {e}")
        raise

def get_sandbox_merged_dir() -> str:
    """
    Get the path to the merged directory of the sandbox.

    Returns:
        str: Path to the merged directory.
    """
    print(f"[DEBUG] Returning sandbox merged dir: {MERGED_DIR}")
    return MERGED_DIR
