import asyncio
import time
import logging
import signal
from watchfiles import awatch
from scripts.generate_code_data import generate_main  

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

# cool down time
THROTTLE_SECONDS = 2
last_run = 0

# Run the generate 
async def generate():
    logging.info("[Watcher] Running generate...")
    await asyncio.to_thread(generate_main)

# Watch for changes and trigger 
async def watch_and_generate():
    global last_run
    async for changes in awatch("data"):
        now = time.time()
        if now - last_run >= THROTTLE_SECONDS:
            logging.info(f"[Watcher] Detected changes: {changes}")
            await generate()
            last_run = now
        else:
            logging.info("[Watcher] Change detected but throttled.")

# able to shut down using Ctrl C
def setup_signal_handlers():
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))

async def shutdown():
    logging.info("[Main] Shutting down watcher...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    await asyncio.gather(*tasks, return_exceptions=True)
    asyncio.get_event_loop().stop()


# Main
async def main():
    logging.info("[Main] Starting file watcher...")
    setup_signal_handlers()
    asyncio.create_task(watch_and_generate())
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
