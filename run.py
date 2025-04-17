import asyncio
import os
import shutil
import socket
from asyncio.subprocess import Process
from pathlib import Path
from shutil import which
from subprocess import CalledProcessError, run
import uvicorn

import dotenv
import rich

dotenv.load_dotenv()


FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", ".frontend"))
APP_HOST = os.getenv("APP_HOST", "localhost")
APP_PORT = int(
    os.getenv("APP_PORT", 8000)
)  # Allocated to backend but also for access to the app, please change it in .env
DEFAULT_FRONTEND_PORT = (
    3000  # Not for access directly, but for proxying to the backend in development
)
STATIC_DIR = Path(os.getenv("STATIC_DIR", "static"))


class NodePackageManager(str):
    def __new__(cls, value: str) -> "NodePackageManager":
        return super().__new__(cls, value)

    @property
    def name(self) -> str:
        return Path(self).stem

    @property
    def is_pnpm(self) -> bool:
        return self.name == "pnpm"

    @property
    def is_npm(self) -> bool:
        return self.name == "npm"


def build():
    """
    Build the frontend and copy the static files to the backend.

    Raises:
        SystemError: If any build step fails
    """
    static_dir = Path("static")

    try:
        package_manager = _get_node_package_manager()
        _install_frontend_dependencies()

        rich.print("\n[bold]Building the frontend[/bold]")
        run([package_manager, "run", "build"], cwd=FRONTEND_DIR, check=True)

        if static_dir.exists():
            shutil.rmtree(static_dir)
        static_dir.mkdir(exist_ok=True)

        shutil.copytree(FRONTEND_DIR / "out", static_dir, dirs_exist_ok=True)

        rich.print(
            "\n[bold]Built frontend successfully![/bold]"
            "\n[bold]Don't forget to update the .env file![/bold]"
        )
    except CalledProcessError as e:
        raise SystemError(f"Build failed during {e.cmd}") from e
    except Exception as e:
        raise SystemError(f"Build failed: {str(e)}") from e


def dev():
    asyncio.run(start_development_servers())


async def start_development_servers():
    """
    Start both frontend and backend development servers.
    Frontend runs with hot reloading, backend runs FastAPI server.

    Raises:
        SystemError: If either server fails to start
    """
    rich.print("\n[bold]Starting development servers[/bold]")
    backend_process = None # Initialize backend_process
    frontend_process = None # Initialize frontend_process

    try:
        processes = []
        if _is_frontend_included():
            frontend_process, frontend_port = await _run_frontend(timeout=90) # Increased timeout to 90 seconds
            processes.append(frontend_process)
            backend_process = await _run_backend(
                envs={
                    "ENVIRONMENT": "dev",
                    "FRONTEND_ENDPOINT": f"http://localhost:{frontend_port}",
                },
            )
            processes.append(backend_process)
        else:
            backend_process = await _run_backend(
                envs={"ENVIRONMENT": "dev"},
            )
            processes.append(backend_process)

        try:
            # Wait for processes to complete
            await asyncio.gather(*[process.wait() for process in processes])
        except (asyncio.CancelledError, KeyboardInterrupt):
            rich.print("\n[bold yellow]Shutting down...[/bold yellow]")
        finally:
            # Terminate both processes
            if frontend_process:
                frontend_process.terminate()
            if backend_process:
                backend_process.terminate()

            # Wait for termination with timeout
            tasks = []
            if frontend_process:
                tasks.append(asyncio.wait_for(frontend_process.wait(), timeout=5))
            if backend_process:
                tasks.append(asyncio.wait_for(backend_process.wait(), timeout=5))

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, asyncio.TimeoutError):
                        process_to_kill = frontend_process if i == 0 and frontend_process else backend_process
                        if process_to_kill:
                            rich.print(f"[bold yellow]Process {process_to_kill.pid} did not terminate gracefully, killing.[/bold yellow]")
                            process_to_kill.kill()

    except Exception as e:
        # Ensure processes are cleaned up even if startup fails
        if frontend_process and frontend_process.returncode is None:
            frontend_process.terminate()
            try: await asyncio.wait_for(frontend_process.wait(), timeout=2)
            except asyncio.TimeoutError: frontend_process.kill()
        if backend_process and backend_process.returncode is None:
            backend_process.terminate()
            try: await asyncio.wait_for(backend_process.wait(), timeout=2)
            except asyncio.TimeoutError: backend_process.kill()
        raise SystemError(f"Failed to start development servers: {str(e)}") from e


async def _run_frontend(
    port: int = DEFAULT_FRONTEND_PORT,
    timeout: int = 90, # Default timeout increased here as well
) -> tuple[Process, int]:
    """
    Start the frontend development server and return its process and port.

    Returns:
        tuple[Process, int]: The frontend process and the port it's running on
    """
    # Install dependencies
    _install_frontend_dependencies()

    port = _find_free_port(start_port=DEFAULT_FRONTEND_PORT)
    package_manager = _get_node_package_manager()

    rich.print(f"\n[bold]Attempting to start frontend on port {port} using {package_manager.name}...[/bold]")
    frontend_process = await asyncio.create_subprocess_exec(
        package_manager,
        "run",
        "dev",
        "--" if package_manager.is_npm else "",
        "-p",
        str(port),
        cwd=FRONTEND_DIR,
        # Capture stderr to help diagnose startup issues
        stderr=asyncio.subprocess.PIPE,
    )
    rich.print(f"\n[bold]Waiting up to {timeout} seconds for frontend (PID: {frontend_process.pid}) to start...")

    start_time = asyncio.get_event_loop().time()
    while (asyncio.get_event_loop().time() - start_time) < timeout:
        await asyncio.sleep(2) # Check every 2 seconds
        if frontend_process.returncode is not None:
            # Read stderr if process exited
            stderr_output = ""
            if frontend_process.stderr:
                 stderr_bytes = await frontend_process.stderr.read()
                 stderr_output = stderr_bytes.decode().strip()
            rich.print(f"[bold red]Frontend process exited unexpectedly with code: {frontend_process.returncode}[/bold red]")
            if stderr_output:
                 rich.print(f"[red]Stderr:\n{stderr_output}[/red]")
            raise RuntimeError("Could not start frontend dev server")
        if _is_server_running(port):
            rich.print(
                f"\n[bold green]Frontend dev server detected as running on port {port} after {asyncio.get_event_loop().time() - start_time:.1f} seconds.[/bold green]"
            )
            # Give it a tiny bit more time just in case
            await asyncio.sleep(1)
            return frontend_process, port

    # If loop finishes, timeout occurred
    frontend_process.terminate()
    try:
        await asyncio.wait_for(frontend_process.wait(), timeout=5)
    except asyncio.TimeoutError:
        frontend_process.kill()
    raise TimeoutError(f"Frontend dev server failed to start on port {port} within {timeout} seconds")


async def _run_backend(envs: dict[str, str | None] = {}) -> Process:
    """Start the backend development server."""
    # Merge environment variables
    envs = {**os.environ, **(envs or {})} # Correctly named 'envs'

    if not _is_port_available(APP_PORT):
        raise SystemError(
            f"Port {APP_PORT} is not available! Please change the port in .env file."
        )

    rich.print(f"\n[bold]Starting backend on port {APP_PORT}...[/bold]")

    # Run uvicorn as a subprocess instead of blocking
    backend_process = await asyncio.create_subprocess_exec(
        "uvicorn",
        "main:app",
        "--host",
        APP_HOST,
        "--port",
        str(APP_PORT),
        "--reload",
        # --- Start Change 2 ---
        # Correct variable name used for environment
        env=envs,
        # --- End Change 2 ---
        # Capture stderr to help diagnose startup issues
        stderr=asyncio.subprocess.PIPE,
    )

    # Wait for port to start
    timeout = 30
    start_time = asyncio.get_event_loop().time()
    while (asyncio.get_event_loop().time() - start_time) < timeout:
        await asyncio.sleep(1)
        # Check the subprocess return code
        if backend_process.returncode is not None:
            # Read stderr if process exited
            stderr_output = ""
            if backend_process.stderr:
                 stderr_bytes = await backend_process.stderr.read()
                 stderr_output = stderr_bytes.decode().strip()
            rich.print(f"[bold red]Backend process exited unexpectedly with code: {backend_process.returncode}[/bold red]")
            if stderr_output:
                 rich.print(f"[red]Stderr:\n{stderr_output}[/red]")
            raise RuntimeError("Could not start backend dev server")
        if _is_server_running(APP_PORT):
            rich.print(
                f"\n[bold green]Backend running at http://{APP_HOST}:{APP_PORT}[/bold green]"
            )
            # Return the process object as intended
            return backend_process

    # If the loop finishes without the server running, terminate the process
    backend_process.terminate()
    stderr_output = ""
    if backend_process.stderr:
        stderr_bytes = await backend_process.stderr.read()
        stderr_output = stderr_bytes.decode().strip()
    try:
        await asyncio.wait_for(backend_process.wait(), timeout=5)
    except asyncio.TimeoutError:
        backend_process.kill() # Force kill if termination fails
    rich.print(f"[bold red]Backend failed to start within {timeout} seconds.[/bold red]")
    if stderr_output:
        rich.print(f"[red]Stderr:\n{stderr_output}[/red]")
    raise TimeoutError(f"Backend failed to start within {timeout} seconds")


def _install_frontend_dependencies():
    package_manager = _get_node_package_manager()
    rich.print(
        f"\n[bold]Installing frontend dependencies using {package_manager.name}. It might take a while...[/bold]"
    )
    run([package_manager, "install"], cwd=".frontend", check=True)


def _get_node_package_manager() -> NodePackageManager:
    """
    Check for available package managers and return the preferred one.
    Returns 'pnpm' if installed, falls back to 'npm'.
    Raises SystemError if neither is installed.

    Returns:
        str: The full path to the available package manager executable
    """
    # On Windows, we need to check for .cmd extensions
    pnpm_cmds = ["pnpm", "pnpm.cmd"]
    npm_cmds = ["npm", "npm.cmd"]

    for cmd in pnpm_cmds:
        cmd_path = which(cmd)
        if cmd_path is not None:
            return NodePackageManager(cmd_path)

    for cmd in npm_cmds:
        cmd_path = which(cmd)
        if cmd_path is not None:
            return NodePackageManager(cmd_path)

    raise SystemError(
        "Neither pnpm nor npm is installed. Please install Node.js and a package manager first."
    )


def _is_port_available(port: int) -> bool:
    """Check if a port is available for binding."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect(("localhost", port))
            return False  # Port is in use, so not available
        except ConnectionRefusedError:
            return True  # Port is available
        except socket.error:
            return True  # Other socket errors likely mean port is available


def _is_server_running(port: int) -> bool:
    """Check if a server is running on the specified port."""
    return not _is_port_available(port)


def _find_free_port(start_port: int) -> int:
    """Find a free port starting from the given port number."""
    for port in range(start_port, 65535):
        if _is_port_available(port):
            return port
    raise SystemError("No free port found")


def _is_frontend_included() -> bool:
    """Check if the app has frontend"""
    return FRONTEND_DIR.exists()

if __name__ == "__main__":
    dev()
