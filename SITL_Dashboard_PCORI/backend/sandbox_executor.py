"""
Sandbox executor for Claude-generated analysis code.

Provides safe execution of Python code in a subprocess with:
- AST-based validation to block dangerous imports
- Process isolation with timeout
- Capture of stdout and matplotlib plots as base64
"""

import ast
import base64
import os
import pickle
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Modules that are blocked for security reasons
BLOCKED_MODULES = {
    # System access
    "os",
    "subprocess",
    "sys",
    "shutil",
    "pathlib",
    # Network access
    "socket",
    "urllib",
    "requests",
    "http",
    "ftplib",
    "smtplib",
    "ssl",
    "asyncio",
    "aiohttp",
    # Code execution
    "pickle",
    "marshal",
    "importlib",
    "builtins",
    "__builtins__",
    # Other dangerous
    "ctypes",
    "multiprocessing",
    "threading",
    "signal",
    "pty",
    "fcntl",
    "resource",
}

# Allowed imports for data analysis
ALLOWED_MODULES = {
    "pandas",
    "pd",
    "numpy",
    "np",
    "matplotlib",
    "matplotlib.pyplot",
    "plt",
    "seaborn",
    "sns",
    "sklearn",
    "scipy",
    "scipy.stats",
    "statsmodels",
    "math",
    "statistics",
    "collections",
    "itertools",
    "functools",
    "operator",
    "typing",
    "dataclasses",
    "datetime",
    "re",
    "json",
    "io",
    "base64",
    "warnings",
}


@dataclass
class ExecutionResult:
    """Result of sandbox code execution."""

    success: bool
    stdout: str
    plot_base64: Optional[str]  # Base64-encoded PNG if a plot was generated
    error: Optional[str]
    execution_time_seconds: float


def validate_code_safety(code: str) -> Tuple[bool, Optional[str]]:
    """
    Validate code using AST to block dangerous operations.

    Returns:
        (is_safe, error_message) - True if safe, False with error message if not
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    for node in ast.walk(tree):
        # Block dangerous imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split(".")[0]
                if module_name in BLOCKED_MODULES:
                    return False, f"Blocked import: {alias.name}"

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_name = node.module.split(".")[0]
                if module_name in BLOCKED_MODULES:
                    return False, f"Blocked import: {node.module}"

        # Block eval/exec calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ("eval", "exec", "compile", "open", "__import__"):
                    return False, f"Blocked function call: {node.func.id}"

        # Block attribute access to dangerous modules
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                if node.value.id in BLOCKED_MODULES:
                    return False, f"Blocked module access: {node.value.id}.{node.attr}"

    return True, None


def prepare_data_for_sandbox(df, y) -> str:
    """
    Serialize DataFrame and target to a temporary pickle file.

    Returns the path to the pickle file.
    """
    # Create temp file that will persist for subprocess
    fd, path = tempfile.mkstemp(suffix=".pkl")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump({"df": df, "y": y}, f)
    except Exception:
        os.close(fd)
        raise
    return path


def execute_in_sandbox(
    code: str,
    df_pickle_path: str,
    timeout: int = 30,
    conda_env: Optional[str] = None,
) -> ExecutionResult:
    """
    Execute code in a subprocess sandbox.

    Args:
        code: Python code to execute
        df_pickle_path: Path to pickled data file
        timeout: Maximum execution time in seconds
        conda_env: Optional conda environment to use

    Returns:
        ExecutionResult with stdout, plot, and any errors
    """
    import time

    start_time = time.time()

    # Build the wrapper script
    wrapper_code = f"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
import pickle

# Set style for dark theme
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#161b22'
plt.rcParams['axes.facecolor'] = '#0d1117'
plt.rcParams['axes.edgecolor'] = '#30363d'
plt.rcParams['axes.labelcolor'] = '#c9d1d9'
plt.rcParams['text.color'] = '#c9d1d9'
plt.rcParams['xtick.color'] = '#8b949e'
plt.rcParams['ytick.color'] = '#8b949e'
plt.rcParams['grid.color'] = '#30363d'

# Load data
with open("{df_pickle_path}", "rb") as f:
    _data = pickle.load(f)
df = _data["df"]
y = _data["y"]

# Optional: make common aliases available
try:
    import seaborn as sns
    sns.set_theme(style="darkgrid")
except ImportError:
    pass

try:
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
except ImportError:
    pass

# Execute user code
{code}

# Capture any open figures
if plt.get_fignums():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='#161b22', edgecolor='none')
    buf.seek(0)
    print("__PLOT_BASE64_START__")
    print(base64.b64encode(buf.getvalue()).decode())
    print("__PLOT_BASE64_END__")
    plt.close('all')
"""

    # Prepare command
    if conda_env:
        cmd = [
            "conda",
            "run",
            "-n",
            conda_env,
            "--no-capture-output",
            sys.executable,
            "-c",
            wrapper_code,
        ]
    else:
        cmd = [sys.executable, "-c", wrapper_code]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),  # Safe working directory
            env={
                **os.environ,
                "PYTHONHASHSEED": "0",  # Reproducibility
            },
        )

        stdout = result.stdout
        stderr = result.stderr
        plot_base64 = None

        # Extract plot if present
        if "__PLOT_BASE64_START__" in stdout:
            parts = stdout.split("__PLOT_BASE64_START__")
            stdout = parts[0]
            plot_part = parts[1].split("__PLOT_BASE64_END__")[0].strip()
            plot_base64 = plot_part

        # Check for errors
        error = None
        if result.returncode != 0:
            error = stderr or "Execution failed with non-zero exit code"
        elif stderr and "Error" in stderr:
            error = stderr

        execution_time = time.time() - start_time

        return ExecutionResult(
            success=result.returncode == 0 and error is None,
            stdout=stdout.strip(),
            plot_base64=plot_base64,
            error=error,
            execution_time_seconds=execution_time,
        )

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=False,
            stdout="",
            plot_base64=None,
            error=f"Execution timed out after {timeout} seconds",
            execution_time_seconds=timeout,
        )
    except Exception as e:
        return ExecutionResult(
            success=False,
            stdout="",
            plot_base64=None,
            error=str(e),
            execution_time_seconds=time.time() - start_time,
        )


def cleanup_pickle_file(path: str) -> None:
    """Remove temporary pickle file."""
    try:
        os.unlink(path)
    except OSError:
        pass


def extract_code_from_response(response_text: str) -> Optional[str]:
    """
    Extract Python code block from Claude's response.

    Looks for ```python ... ``` blocks.
    """
    import re

    # Try to find Python code block
    pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Try generic code block
    pattern = r"```\s*(.*?)\s*```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    if matches:
        return matches[0].strip()

    return None
