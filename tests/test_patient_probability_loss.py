"""Keep the CPU Torch loss checks in a fresh interpreter on macOS."""
from pathlib import Path
import subprocess
import sys


def test_isolated_patient_probability_loss_cases():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', 'tests/cpu_cases/patient_probability_loss_cases.py', '-q'],
        cwd=root, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    assert '3 passed' in result.stdout, result.stdout
