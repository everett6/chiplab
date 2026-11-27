#!/usr/bin/env python3
"""Run the experimental tools and prompts tests."""
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_mcp_experimental_tools.py", "tests/test_prompts.py", "-v"],
    cwd="/home/everett/Chip Lab",
    capture_output=True,
    text=True,
)

print("STDOUT:")
print(result.stdout)
print("STDERR:")
print(result.stderr)
print("Return code:", result.returncode)