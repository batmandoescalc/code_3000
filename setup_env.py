import subprocess
import sys
import venv
from pathlib import Path

ENV_DIR = Path("3000-env")

# Create venv
venv.create(ENV_DIR, with_pip=True)

if sys.platform == "win32":
    pip = ENV_DIR / "Scripts" / "pip.exe"
    python = ENV_DIR / "Scripts" / "python.exe"
else:
    pip = ENV_DIR / "bin" / "pip"
    python = ENV_DIR / "bin" / "python"

packages = [
    "numpy==2.4.6",
    "pandas==2.3.3",
    "scikit-learn==1.8.0",
    "matplotlib==3.11.2",
    "seaborn==0.13.2",
    "shap==0.52.0",
]

subprocess.check_call([pip, "install", "--upgrade", "pip"])
subprocess.check_call([pip, "install", *packages])

subprocess.check_call([
    python, "-c",
    "import numpy, sklearn, shap; print('Setup complete')"
])
