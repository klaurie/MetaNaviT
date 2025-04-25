"""
overlayfs_utils.py

Low-level utilities for managing OverlayFS mounts.
Handles mounting and unmounting of the overlay filesystem.
"""

import subprocess

def mount_overlay(lower: str, upper: str, work: str, merged: str):
    print(f"[DEBUG] Mounting OverlayFS:")
    print(f"  lowerdir={lower}")
    print(f"  upperdir={upper}")
    print(f"  workdir={work}")
    print(f"  merged={merged}")
    
    cmd = [
        "mount", "-t", "overlay", "overlay",
        "-o", f"lowerdir={lower},upperdir={upper},workdir={work}",
        merged
    ]
    subprocess.run(cmd, check=True)


def unmount_overlay(mount_point: str):
    print(f"[DEBUG] Unmounting OverlayFS at: {mount_point}")
    subprocess.run(["umount", mount_point], check=True)
