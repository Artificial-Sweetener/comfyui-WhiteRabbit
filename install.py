import os
import subprocess
import sys
import importlib.util

def is_installed(package_name):
    spec = importlib.util.find_spec(package_name)
    return spec is not None

package_to_check = "packaging"

if not is_installed(package_to_check):
    try:
        print(f"Installing missing package: {package_to_check}")
        subprocess.run([sys.executable, "-m", "pip", "install", package_to_check], check=True)
        print(f"{package_to_check} installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package_to_check}: {e}", file=sys.stderr)
else:
    print(f"{package_to_check} is already installed.")


# Get the path to the custom_nodes directory
custom_nodes_path = os.path.dirname(os.path.abspath(__file__))
comfyui_custom_nodes_path = os.path.dirname(custom_nodes_path)

# The directory to clone into
clone_dir = os.path.join(comfyui_custom_nodes_path, "ComfyUI-Frame-Interpolation")

# Check if the directory already exists
if not os.path.exists(clone_dir):
    try:
        print("Cloning ComfyUI-Frame-Interpolation repository...")
        subprocess.run(["git", "clone", "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation", clone_dir], check=True)
        print("Cloned ComfyUI-Frame-Interpolation repository successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}", file=sys.stderr)
        sys.exit(1)
else:
    print("ComfyUI-Frame-Interpolation repository already exists.")