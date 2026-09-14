from __future__ import annotations

import runpy
import sys

sys.argv.extend(["--backend", "vector"])
runpy.run_module("src.backend.common.run_official_experiment", run_name="__main__")

