"""Tests for async file watcher behavior with throttling and task delegation"""

import asyncio
import pytest
from unittest import mock
from main import generate, watch_and_generate

@pytest.mark.asyncio
@mock.patch("main.generate_main")
@mock.patch("asyncio.to_thread")
async def test_generate_triggers_generate_main(mock_to_thread, mock_generate_main):
    """Test that generate() correctly calls generate_main via asyncio.to_thread."""
    await generate()
    mock_to_thread.assert_called_once_with(mock_generate_main)

@pytest.mark.asyncio
@mock.patch("main.awatch")
@mock.patch("main.generate")
async def test_watch_and_generate_respects_throttle(mock_generate, mock_awatch):
    """Test that generate() is not called too frequently due to throttle cooldown."""
    # Simulate two changes in quick succession (1 second apart, under the 2s throttle)
    mock_awatch.return_value.__aiter__.return_value = [
        {("data/file1.txt", 1)},
        {("data/file2.txt", 1)},
    ]

    # Mock time to simulate throttling window
    with mock.patch("main.time") as mock_time:
        mock_time.time.side_effect = [1000, 1000, 1001]

        task = asyncio.create_task(watch_and_generate())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert mock_generate.call_count == 1, "generate() should only be called once due to throttling"


@pytest.mark.asyncio
@mock.patch("main.asyncio.get_event_loop")
@mock.patch("main.asyncio.all_tasks")
async def test_shutdown_cancels_tasks_and_stops_loop(mock_all_tasks, mock_get_loop):
    """Test that shutdown cancels all other tasks and stops the loop."""
    from main import shutdown

    dummy_task = mock.AsyncMock()
    dummy_task.cancel = mock.Mock()
    mock_all_tasks.return_value = [dummy_task]
    mock_get_loop.return_value = mock.Mock()

    await shutdown()

    dummy_task.cancel.assert_called_once()
    mock_get_loop.return_value.stop.assert_called_once()


@mock.patch("asyncio.get_event_loop")
def test_setup_signal_handlers(mock_get_loop):
    """Test that signal handlers are registered for SIGINT and SIGTERM."""
    from main import setup_signal_handlers
    mock_loop = mock.Mock()
    mock_get_loop.return_value = mock_loop

    setup_signal_handlers()

    calls = [mock.call(signal.SIGINT, mock.ANY), mock.call(signal.SIGTERM, mock.ANY)]
    mock_loop.add_signal_handler.assert_has_calls(calls, any_order=True)
