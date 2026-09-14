from __future__ import annotations

import runpy
import sys

sys.argv.extend(["--backend", "lightrag"])
runpy.run_module("scripts.run_official_experiment", run_name="__main__")

