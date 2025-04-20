"""
dry_run_manager.py

Handles tracking and toggling of dry-run mode.
This will later support approval-based transitions from dry-run to real execution.
"""

class DryRunManager:
    _active = False

    @classmethod
    def activate(cls):
        cls._active = True

    @classmethod
    def deactivate(cls):
        cls._active = False

    @classmethod
    def is_active(cls):
        return cls._active
