"""
Static GA versus the classical population-based metaheuristics.

Runs the static GA and any of

* ``de``  - differential evolution (DE/rand/1/bin),
* ``pso`` - particle swarm optimisation,
* ``abc`` - artificial bee colony,
* ``css`` - charged system search,

on the same instances, from the same initial populations, against the same
objective and under the same evaluation budget, then writes one ``runs.csv``
that the existing analysis and figure code turns into a joint report.

What "the same starting point" means here
-----------------------------------------
Every (instance, seed) pair has one run seed, ``42 + 100 * instance_index +
run``, and one initial population built from it by
:func:`optimization.ga_core.make_initial_population`.  The GA and all four
baselines are handed *that* population, so on a given pair they all start from
a bit-identical set of 200 layouts and differ only in what they do next.  That
is also what makes the paired statistics (Wilcoxon on matched pairs) valid.
``tests/test_metaheuristics.py`` asserts the property rather than trusting it.

Budget
------
The GA without local search spends ``population_size * generations`` = 60 000
objective evaluations, and every baseline is capped at the same number.  DE,
PSO and CSS spend one population per iteration, so they run the same 300
iterations; ABC spends two (employed + onlooker bees), so it runs about 150.
The analysis additionally compares every method at an equal number of
evaluations and at equal wall-clock time.

Parameters
----------
The baselines run at their textbook defaults (``--list-parameters`` prints
them, ``BASELINES.md`` gives the sources); each is overridable from the
command line.  The GA operators are the same flags ``run_static_ga`` takes.

Stages
------
``instances`` -> ``main`` -> ``timing`` -> ``analyze`` -> ``figures``, exactly
as in :mod:`run_static_ga`; ``--only`` and ``--skip`` select among them.

Examples
--------
Everything, defaults (10 instances x 10 seeds x 5 methods)::

    python -m run_baselines

Two baselines against two GA settings, 5 seeds::

    python -m run_baselines --algorithms de pso --ga-variants no-ls ls100-elites --runs 5

A quick look at one instance::

    python -m run_baselines --instances 0 --runs 2 --only main analyze

Add the baselines to an existing static-GA results directory instead::

    python -m run_baselines --ga-variants none --results-dir results/static_ga --append
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import replace
from typing import Dict, Optional, Sequence, Union

from optimization.ga_core import DEFAULT_LS_SCHEDULE, GAConfig
from optimization.metaheuristics import ALGORITHM_NAMES, ALGORITHMS, MetaConfig
from optimization.run_experiments import read_rows
from paths import RESULTS_ROOT
from run_static_ga import (
    CATALOGUE,
    GROUPS,
    STAGES,
    _positive_int,
    _unit_interval,
    build_configs,
    parse_schedule,
    run_static_grid,
)

#: Default output location: its own directory, so neither the archived RL-GA
#: study nor the static-GA one is touched.
DEFAULT_BASELINE_RESULTS_DIR = os.path.join(RESULTS_ROOT, "baselines")

#: GA variants run alongside the baselines unless ``--ga-variants`` says
#: otherwise.  The plain GA is the one the comparison is about; the local
#: search variants are available by name for a wider grid.
DEFAULT_GA_VARIANTS: Sequence[str] = ("no-ls",)


def variant_key(algorithm: str, tag: Optional[str]) -> str:
    return f"{algorithm}_{tag}" if tag else algorithm


def build_meta_configs(
    algorithms: Sequence[str],
    base: MetaConfig,
    tag: Optional[str],
) -> Dict[str, MetaConfig]:
    """``["de", "pso"]`` -> ``{variant key: MetaConfig}``."""
    configs: Dict[str, MetaConfig] = {}
    for algorithm in algorithms:
        if algorithm not in ALGORITHMS:
            raise SystemExit(
                f"unknown algorithm {algorithm!r}. Choose from {', '.join(ALGORITHMS)}."
            )
        label = ALGORITHM_NAMES[algorithm] + (f" ({tag})" if tag else "")
        configs[variant_key(algorithm, tag)] = replace(base, algorithm=algorithm, label=label)
    return configs


def print_parameters(config: MetaConfig) -> None:
    """Print the baseline parameter defaults and where they come from."""
    print("Baseline algorithms and their default parameters.\n")
    blocks = [
        (
            "de",
            "DE/rand/1/bin (Storn & Price, J. Global Optim. 1997)",
            [
                ("--de-f", f"{config.de_f}", "differential weight F"),
                ("--de-cr", f"{config.de_cr}", "crossover probability CR"),
            ],
        ),
        (
            "pso",
            "global-best PSO, constriction (Clerc & Kennedy, IEEE TEC 2002)",
            [
                ("--pso-w", f"{config.pso_w}", "inertia / constriction factor"),
                ("--pso-c1", f"{config.pso_c1}", "cognitive coefficient"),
                ("--pso-c2", f"{config.pso_c2}", "social coefficient"),
                (
                    "--pso-vmax-fraction",
                    f"{config.pso_vmax_fraction}",
                    "velocity clamp as a fraction of each variable's range",
                ),
            ],
        ),
        (
            "abc",
            "ABC (Karaboga, Erciyes Univ. TR06, 2005)",
            [
                ("(SN)", "population size", "food sources; the colony is 2 x SN bees"),
                ("--abc-limit", "SN x D", "abandonment threshold"),
            ],
        ),
        (
            "css",
            "CSS (Kaveh & Talatahari, Acta Mech. 2010)",
            [
                ("--css-cm-fraction", f"{config.css_cm_fraction}", "charged memory size / CP"),
                ("--css-ka", f"{config.css_ka}", "acceleration coefficient base"),
                ("--css-kv", f"{config.css_kv}", "velocity coefficient base"),
                ("--css-a-fraction", f"{config.css_a_fraction}", "radius a, normalised units"),
                ("--css-cmcr", f"{config.css_cmcr}", "harmony-search boundary handling"),
                ("--css-par", f"{config.css_par}", "pitch adjustment rate"),
            ],
        ),
    ]
    for key, source, rows in blocks:
        print(f"  {key} - {source}")
        for flag, value, description in rows:
            print(f"      {flag:<22} {value:<16} {description}")
        print()
    print("Population size, iteration cap and evaluation budget are shared with the GA")
    print("(--population-size, --generations); see --help.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_baselines",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    stages = parser.add_argument_group("stages")
    stages.add_argument("--only", nargs="+", choices=list(STAGES), default=None,
                        help="run only these stages (default: all of them)")
    stages.add_argument("--skip", nargs="+", choices=list(STAGES), default=(),
                        help="run everything except these stages")

    grid = parser.add_argument_group("what to run")
    grid.add_argument(
        "--algorithms",
        nargs="+",
        default=list(ALGORITHMS),
        metavar="NAME",
        help=f"baselines to run: {', '.join(ALGORITHMS)}, or 'none' "
        f"(default: all four)",
    )
    grid.add_argument(
        "--ga-variants",
        nargs="+",
        default=list(DEFAULT_GA_VARIANTS),
        metavar="NAME",
        help="static GA variants to run alongside them, or 'none' to compare "
        "against GA rows already in the results directory (default: no-ls). "
        "Accepts the same names as run_static_ga --variants",
    )
    grid.add_argument("--list-parameters", action="store_true",
                      help="print the baseline parameter defaults and exit")
    grid.add_argument("--split", default="test", choices=("train", "test"),
                      help="instance split to run on (default: test)")
    grid.add_argument("--instances", type=int, nargs="+", default=None, metavar="INDEX",
                      help="restrict to these instance indices (default: all of them)")
    grid.add_argument("--runs", type=_positive_int("--runs"), default=10,
                      help="independent seeds per instance in the main grid (default: 10)")
    grid.add_argument("--timing-runs", type=_positive_int("--timing-runs"), default=3,
                      help="seeds per instance in the sequential timing pass (default: 3)")
    grid.add_argument(
        "--workers",
        type=_positive_int("--workers"),
        default=max(1, (os.cpu_count() or 4) - 2),
        help="parallel worker processes for the main grid; the timing stage always "
        "runs sequentially (default: cpu_count - 2)",
    )

    budget = parser.add_argument_group("budget (shared by every method)")
    budget.add_argument("--population-size", type=_positive_int("--population-size"), default=200,
                        help="population / swarm / colony size (default: 200)")
    budget.add_argument("--generations", type=_positive_int("--generations"), default=300,
                        help="iteration cap (default: 300)")
    budget.add_argument(
        "--max-evaluations",
        type=_positive_int("--max-evaluations"),
        default=None,
        help="objective-evaluation cap per run (default: population-size x generations, "
        "i.e. the budget the GA without local search spends)",
    )

    operators = parser.add_argument_group("GA operators")
    operators.add_argument("--tournament-k", type=_positive_int("--tournament-k"), default=3,
                           help="tournament selection size (default: 3)")
    operators.add_argument("--elitism-rate", type=_unit_interval("--elitism-rate", upper_open=True),
                           default=0.1, help="default: 0.1")
    operators.add_argument("--crossover-rate", type=_unit_interval("--crossover-rate"), default=0.8,
                           help="default: 0.8")
    operators.add_argument("--mutation-rate", type=_unit_interval("--mutation-rate"), default=0.1,
                           help="default: 0.1")
    operators.add_argument("--ls-iterations", type=_positive_int("--ls-iterations"), default=5,
                           help="local-search steps per selected individual (default: 5)")
    operators.add_argument("--ls-rate", type=_unit_interval("--ls-rate"), default=None,
                           help="local-search rate for the GA 'custom' variant")
    operators.add_argument("--ls-target", default="elites", choices=("offspring", "elites"),
                           help="where the GA 'custom' variant applies local search")
    operators.add_argument("--ls-schedule", nargs="+", default=None, metavar="GEN:RATE",
                           help="schedule for the deterministic GA variants, e.g. 0:0.1 100:0.5")

    de = parser.add_argument_group("differential evolution")
    de.add_argument("--de-f", type=float, default=None, help="differential weight F (default: 0.5)")
    de.add_argument("--de-cr", type=_unit_interval("--de-cr"), default=None,
                    help="crossover probability CR (default: 0.9)")

    pso = parser.add_argument_group("particle swarm")
    pso.add_argument("--pso-w", type=float, default=None, help="inertia weight (default: 0.7298)")
    pso.add_argument("--pso-c1", type=float, default=None, help="cognitive coefficient (default: 1.49618)")
    pso.add_argument("--pso-c2", type=float, default=None, help="social coefficient (default: 1.49618)")
    pso.add_argument("--pso-vmax-fraction", type=_unit_interval("--pso-vmax-fraction"), default=None,
                     help="velocity clamp as a fraction of each variable's range (default: 0.5)")

    abc = parser.add_argument_group(
        "artificial bee colony (SN = --population-size: the food sources are the "
        "shared initial population, which is what keeps ABC's starting point "
        "identical to every other method's)"
    )
    abc.add_argument("--abc-limit", type=_positive_int("--abc-limit"), default=None,
                     help="abandonment threshold (default: SN x D)")

    css = parser.add_argument_group("charged system search")
    css.add_argument("--css-cm-fraction", type=_unit_interval("--css-cm-fraction"), default=None,
                     help="charged memory size as a fraction of CP (default: 0.25)")
    css.add_argument("--css-ka", type=float, default=None, help="acceleration coefficient (default: 0.5)")
    css.add_argument("--css-kv", type=float, default=None, help="velocity coefficient (default: 0.5)")
    css.add_argument("--css-a-fraction", type=float, default=None,
                     help="radius a in normalised units (default: 0.10)")
    css.add_argument("--css-cmcr", type=_unit_interval("--css-cmcr"), default=None,
                     help="harmony-search boundary handling rate (default: 0.95)")
    css.add_argument("--css-par", type=_unit_interval("--css-par"), default=None,
                     help="pitch adjustment rate (default: 0.10)")
    css.add_argument("--css-mean-force", action="store_true", default=None,
                     help="average the electric force over the sources instead of "
                     "summing it as the paper does (default: off; see BASELINES.md)")

    output = parser.add_argument_group("output")
    output.add_argument("--results-dir", default=DEFAULT_BASELINE_RESULTS_DIR,
                        help=f"default: {DEFAULT_BASELINE_RESULTS_DIR}")
    output.add_argument("--tag", default=None,
                        help="suffix added to every variant key and label, so a second "
                        "invocation stacks alongside the first instead of replacing it")
    output.add_argument("--append", action="store_true",
                        help="merge into an existing runs.csv instead of replacing it")
    output.add_argument("--no-trace", action="store_true",
                        help="skip storing per-iteration traces (the convergence and "
                        "budget-matched analyses need them)")
    output.add_argument("--reference", default=None,
                        help="variant every comparison is measured against "
                        "(default: the GA variant present, so the report reads "
                        "'GA minus baseline')")
    output.add_argument("--baseline", default=None,
                        help="variant the per-instance gain is measured from "
                        "(default: the first baseline algorithm present)")
    return parser


def meta_config_from_args(args: argparse.Namespace) -> MetaConfig:
    """The shared baseline configuration, with any command-line overrides."""
    overrides = {
        field: getattr(args, field)
        for field in (
            "de_f", "de_cr",
            "pso_w", "pso_c1", "pso_c2", "pso_vmax_fraction",
            "abc_limit",
            "css_cm_fraction", "css_ka", "css_kv", "css_a_fraction", "css_cmcr", "css_par",
            "css_mean_force",
        )
        if getattr(args, field) is not None
    }
    return MetaConfig(
        population_size=args.population_size,
        generations=args.generations,
        max_evaluations=args.max_evaluations,
        **overrides,
    )


def pick_reference(available: Sequence[str], reference: Optional[str], baseline: Optional[str]):
    """Default the report's reference to the GA and its contrast to a baseline.

    The archived analysis was written around an RL-GA reference; here the
    question is "how does the GA compare with each classical metaheuristic",
    so the GA is the reference and the differences the report prints are
    ``GA - baseline`` on matched (instance, seed) pairs.
    """
    if reference and reference not in available:
        raise SystemExit(f"--reference {reference!r} is not in the results. Have: {list(available)}")
    if baseline and baseline not in available:
        raise SystemExit(f"--baseline {baseline!r} is not in the results. Have: {list(available)}")

    def is_baseline(key: str) -> bool:
        return any(re.fullmatch(rf"{algorithm}(?:_[A-Za-z0-9-]+)?", key) for algorithm in ALGORITHMS)

    ga_variants = [key for key in available if not is_baseline(key)]
    # Ordered as ALGORITHMS lists them rather than alphabetically, so the
    # default contrast is stable and does not depend on which methods ran.
    baselines = sorted(
        (key for key in available if is_baseline(key)),
        key=lambda key: next(i for i, a in enumerate(ALGORITHMS) if key.startswith(a)),
    )

    reference = reference or (ga_variants[0] if ga_variants else available[0])
    baseline = baseline or next(
        (key for key in baselines if key != reference),
        next((key for key in available if key != reference), reference),
    )
    return reference, baseline


def main() -> None:
    args = build_parser().parse_args()

    if args.list_parameters:
        print_parameters(MetaConfig())
        return

    if args.tag is not None and not re.fullmatch(r"[A-Za-z0-9-]+", args.tag):
        raise SystemExit(f"--tag may only contain letters, digits and dashes, got {args.tag!r}")
    if args.tournament_k > args.population_size:
        raise SystemExit(
            f"--tournament-k ({args.tournament_k}) cannot exceed --population-size "
            f"({args.population_size})."
        )

    algorithms = [] if args.algorithms == ["none"] else list(args.algorithms)
    ga_names = [] if args.ga_variants == ["none"] else list(args.ga_variants)
    unknown = [n for n in ga_names if n not in CATALOGUE and n not in GROUPS and n != "custom"]
    if unknown:
        raise SystemExit(
            f"unknown GA variant(s) {unknown}. Run 'python -m run_static_ga --list-variants'."
        )
    if not algorithms and not ga_names:
        raise SystemExit("nothing to run: --algorithms and --ga-variants are both 'none'.")

    schedule = parse_schedule(args.ls_schedule) if args.ls_schedule else DEFAULT_LS_SCHEDULE
    ga_base = GAConfig(
        population_size=args.population_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        elitism_rate=args.elitism_rate,
        tournament_k=args.tournament_k,
        ls_iterations=args.ls_iterations,
        ls_schedule=schedule,
        max_evaluations=args.max_evaluations,
    )

    configs: Dict[str, Union[GAConfig, MetaConfig]] = {}
    if ga_names:
        configs.update(build_configs(ga_names, ga_base, args.ls_rate, args.ls_target, args.tag))
    if algorithms:
        configs.update(build_meta_configs(algorithms, meta_config_from_args(args), args.tag))

    stages = [s for s in (args.only or STAGES) if s not in args.skip]
    if not stages:
        raise SystemExit("--skip removed every stage; nothing to do.")

    budget = args.max_evaluations or args.population_size * args.generations
    print(f"Baseline comparison pipeline -- stages: {', '.join(stages)}")
    print(f"Results directory : {args.results_dir}")
    if {"main", "timing"} & set(stages):
        print(
            f"Budget            : population {args.population_size}, "
            f"{args.generations} iterations, {budget} evaluations per run"
        )
        print("Methods           :")
        for key, config in configs.items():
            if isinstance(config, MetaConfig):
                print(f"  {key:<26} {ALGORITHM_NAMES[config.algorithm]}")
            else:
                rate = (
                    "scheduled" if config.ls_controller == "schedule" else f"{config.ls_rate:.0%}"
                )
                print(f"  {key:<26} static GA, LS {rate} on {config.ls_target}")
    print()

    started = time.perf_counter()
    for stage in stages:
        print(f"\n{'=' * 78}\n== {stage}\n{'=' * 78}")
        stage_started = time.perf_counter()

        if stage == "instances":
            from core.problems import dump_instance_catalogue

            os.makedirs(args.results_dir, exist_ok=True)
            dump_instance_catalogue(os.path.join(args.results_dir, "instances.json"))

        elif stage == "main":
            run_static_grid(
                configs=configs,
                split=args.split,
                runs=args.runs,
                workers=args.workers,
                output_dir=args.results_dir,
                instance_indices=args.instances,
                csv_name="runs.csv",
                store_trace=not args.no_trace,
                append=args.append,
                pipeline="baselines",
            )

        elif stage == "timing":
            print(
                "Sequential timing replication (one run at a time) so the wall-clock "
                "comparison is not distorted by parallel load."
            )
            run_static_grid(
                configs=configs,
                split=args.split,
                runs=args.timing_runs,
                workers=1,
                output_dir=args.results_dir,
                instance_indices=args.instances,
                csv_name="runs_timing.csv",
                store_trace=False,
                append=args.append,
                pipeline="baselines",
            )

        elif stage in ("analyze", "figures"):
            csv_path = os.path.join(args.results_dir, "runs.csv")
            if not os.path.exists(csv_path):
                raise SystemExit(
                    f"{csv_path} does not exist -- run the 'main' stage first, or point "
                    f"--results-dir at a directory that already has one."
                )
            available = sorted({row["variant"] for row in read_rows(csv_path)})
            reference, baseline = pick_reference(available, args.reference, args.baseline)
            print(f"Reference variant: {reference}   contrast: {baseline}")

            if stage == "analyze":
                from analysis.analyze import analyse

                analyse(
                    results_dir=args.results_dir,
                    csv_name="runs.csv",
                    split=args.split,
                    references=[reference],
                    baseline=baseline,
                )
            else:
                from visualization.plots import make_all_figures

                make_all_figures(
                    results_dir=args.results_dir,
                    csv_name="runs.csv",
                    split=args.split,
                    reference=reference,
                    baseline=baseline,
                )

        print(f"-- {stage} finished in {(time.perf_counter() - stage_started) / 60:.1f} min")

    print(f"\nBaseline comparison finished in {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
