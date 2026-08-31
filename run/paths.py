"""Where this project reads and writes, resolved once and absolutely.

Every default output location is derived from the position of this file, so a
stage behaves the same whether it is launched from the repository root, from
a subdirectory, or from a multiprocessing worker that inherited a different
working directory.  Nothing here reads the environment; the entry points all
accept explicit overrides (``--results-dir``, ``--agent-dir``, ``--bins``)
and pass them down.
"""

from __future__ import annotations

import os

#: This package's directory (whatever the containing folder is named).
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))

#: The repository checkout that contains it.
REPO_ROOT = os.path.dirname(PACKAGE_ROOT)

#: Everything a run produces lives under here.  Each study owns one
#: subdirectory of it, so no two studies can overwrite each other.
RESULTS_ROOT = os.path.join(PACKAGE_ROOT, "results")

#: The archived RL-GA study: ``runs.csv``, ``raw/``, ``analysis/``,
#: ``figures/``, ``agents/`` and the static-GA sub-study.  Re-running without
#: ``--results-dir`` overwrites it in place.
DEFAULT_RESULTS_DIR = os.path.join(RESULTS_ROOT, "results_rl_ga")

#: The DE / PSO / ABC / CSS comparison written by ``run.run_baselines``.
DEFAULT_BASELINE_RESULTS_DIR = os.path.join(RESULTS_ROOT, "results_baselines")

#: The GA crossover x mutation grid written by ``run.run_ablation``.
DEFAULT_ABLATION_RESULTS_DIR = os.path.join(RESULTS_ROOT, "results_ablation")

#: The PSO c1 / c2 study written by ``run.run_pso_ablation``.
DEFAULT_PSO_ABLATION_RESULTS_DIR = os.path.join(RESULTS_ROOT, "results_pso_ablation")

#: Trained Q-tables, training histories and the calibrated state bins.
#: Nested under ``DEFAULT_RESULTS_DIR`` so the two move together; pass
#: ``--agent-dir`` to point somewhere else.
DEFAULT_AGENT_DIR = os.path.join(DEFAULT_RESULTS_DIR, "agents")

#: State-discretisation thresholds calibrated on the training split.
DEFAULT_BINS_PATH = os.path.join(DEFAULT_AGENT_DIR, "state_bins.json")

#: Figures produced by the standalone scripts in ``visualization/extra/``.
EXTRA_FIGURES_DIR = os.path.join(DEFAULT_RESULTS_DIR, "figures", "extra")
