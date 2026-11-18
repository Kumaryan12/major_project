"""Exercise CPU embedding cases in a fresh interpreter, matching production workers.

The combined macOS numerical runtime segfaulted inside Conv1d when this module
followed other feature/kernel tests. Fresh-process cases and the real-data CLI
smoke pass. Isolation avoids sharing mutable Torch/native runtime state; failures
remain fatal and no numerical libraries, integrity checks or overrides change.
"""
from pathlib import Path
import subprocess
import sys


def test_isolated_learned_distribution_cases():
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'-m','pytest','tests/cpu_cases/learned_distribution_cases.py','-q','--disable-warnings'],
                          cwd=root,capture_output=True,text=True,timeout=240)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    assert '9 passed' in result.stdout,result.stdout
