"""Run inherited economy contracts and integration tests in isolated processes."""

import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, CAMBC_TEST_BOT="final")
    for modules in (
        ["tests.nexus.test_cache_and_planning", "tests.nexus.test_planning_slices"],
        ["tests.final.test_integration"],
    ):
        subprocess.run([sys.executable, "-m", "unittest", *modules, "-v"],
                       cwd=root, env=env, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
