"""
overlayfs_utils.py

Low-level utilities for managing OverlayFS mounts.
Handles mounting and unmounting of the overlay filesystem.
"""

import subprocess

def mount_overlay(lower: str, upper: str, work: str, merged: str):
    cmd = [
        "mount", "-t", "overlay", "overlay",
        "-o", f"lowerdir={lower},upperdir={upper},workdir={work}",
        merged
    ]
    subprocess.run(cmd, check=True)

def unmount_overlay(mount_point: str):
    subprocess.run(["umount", mount_point], check=True)
