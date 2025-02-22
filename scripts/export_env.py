#!/usr/bin/env python3
"""
Development Environment Export Script

Exports the current development environment configuration including:
1. Python dependencies
2. Node.js dependencies
3. Environment variables
4. System dependencies
5. Database configuration
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, Any

def export_python_dependencies() -> Dict[str, Any]:
    """Export Python package dependencies from poetry"""
    try:
        # Get dependencies from pyproject.toml
        pyproject_path = Path("pyproject.toml")
        if not pyproject_path.exists():
            print("No pyproject.toml found")
            return {}

        # Use poetry show for dependency list
        dependencies = subprocess.check_output(
            ["poetry", "show", "--tree"], 
            text=True
        )
        dev_dependencies = subprocess.check_output(
            ["poetry", "show", "--only", "dev", "--tree"],
            text=True
        )
        
        # Get Python version from poetry env
        python_version = subprocess.check_output(
            ["poetry", "env", "info", "--path"],
            text=True
        ).strip()

        return {
            "dependencies": dependencies,
            "dev_dependencies": dev_dependencies,
            "python_version": python_version,
            "pyproject": pyproject_path.read_text()
        }
    except subprocess.CalledProcessError as e:
        print(f"Failed to export Python dependencies: {e}")
        return {}

def export_node_dependencies() -> Dict[str, Any]:
    """Export Node.js dependencies from package.json"""
    frontend_dir = Path(".frontend")
    if not frontend_dir.exists():
        return {}
    
    package_json = frontend_dir / "package.json"
    if (package_json.exists()):
        return json.loads(package_json.read_text())
    return {}

def export_env_variables() -> Dict[str, str]:
    """Export relevant environment variables"""
    relevant_vars = [
        "MODEL_PROVIDER",
        "MODEL",
        "EMBEDDING_MODEL",
        "PG_CONNECTION_STRING",
        "ENVIRONMENT",
        "APP_HOST",
        "APP_PORT",
        "FRONTEND_DIR",
        "STATIC_DIR",
        "DATA_DIR",
        "STORAGE_DIR"
    ]
    
    return {
        var: os.getenv(var, "") 
        for var in relevant_vars 
        if os.getenv(var)
    }

def export_system_info() -> Dict[str, Any]:
    """Export system dependencies and versions"""
    info = {}
    
    # Check Poetry version
    try:
        info["poetry"] = subprocess.check_output(
            ["poetry", "--version"], text=True
        ).strip()
    except: pass
    
    # Check Node.js version
    try:
        info["node"] = subprocess.check_output(
            ["node", "--version"], text=True
        ).strip()
    except: pass
    
    # Check npm/pnpm version
    try:
        info["npm"] = subprocess.check_output(
            ["npm", "--version"], text=True
        ).strip()
    except: pass
    
    try:
        info["pnpm"] = subprocess.check_output(
            ["pnpm", "--version"], text=True
        ).strip()
    except: pass
    
    # Check PostgreSQL version
    try:
        info["postgresql"] = subprocess.check_output(
            ["psql", "--version"], text=True
        ).strip()
    except: pass
    
    return info

def export_configuration() -> Dict[str, Any]:
    """Export all configuration files"""
    config_dir = Path("config")
    if not config_dir.exists():
        return {}
    
    configs = {}
    for config_file in config_dir.glob("*.yaml"):
        configs[config_file.name] = config_file.read_text()
    return configs

def main():
    """Export full environment configuration"""
    try:
        export = {
            "python": export_python_dependencies(),
            "node": export_node_dependencies(),
            "env": export_env_variables(),
            "system": export_system_info(),
            "config": export_configuration()
        }
        
        # Save to file
        output_dir = Path("scripts/env_exports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "environment_export.json"
        with open(output_file, "w") as f:
            json.dump(export, f, indent=2)
        
        print(f"Environment exported to {output_file}")
        
        # Create setup script only if we have data
        if any(export.values()):
            generate_setup_script(output_dir, export)
        else:
            print("Warning: No environment data was exported")
            
    except Exception as e:
        print(f"Export failed: {e}")
        raise

def generate_setup_script(output_dir: Path, export: Dict[str, Any]) -> None:
    """Generate setup script from exported data"""
    setup_script = output_dir / "setup_from_export.sh"
    with open(setup_script, "w") as f:
        f.write(f"""#!/bin/bash
set -e  # Exit on error

echo "Setting up development environment..."

# Install system dependencies
{generate_system_install_commands(export['system'])}

# Set up Python environment
if ! command -v poetry &> /dev/null; then
    curl -sSL https://install.python-poetry.org | python3 -
fi
poetry install

# Set up Node.js environment
if [ -d ".frontend" ]; then
    cd .frontend
    npm install
    cd ..
fi

# Set up environment variables
{generate_env_commands(export['env'])}

# Initialize database
if [ -f "setup.py" ]; then
    python setup.py
fi

echo "Environment setup complete!"
""")
    
    setup_script.chmod(0o755)
    print(f"Setup script generated at {setup_script}")

def generate_system_install_commands(system_info: Dict[str, str]) -> str:
    """Generate installation commands for system dependencies"""
    commands = []
    if "postgresql" not in system_info:
        commands.append("sudo apt-get install -y postgresql postgresql-contrib")
    if "node" not in system_info:
        commands.append("curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -")
        commands.append("sudo apt-get install -y nodejs")
    if "poetry" not in system_info:
        commands.append('curl -sSL https://install.python-poetry.org | python3 -')
    return "\n".join(commands)

def generate_env_commands(env_vars: Dict[str, str]) -> str:
    """Generate commands to set environment variables"""
    return "\n".join([
        f'echo "{key}={value}" >> .env'
        for key, value in env_vars.items()
    ])

if __name__ == "__main__":
    main()