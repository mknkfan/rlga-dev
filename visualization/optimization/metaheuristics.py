"""
Population-based baselines: DE, PSO, ABC and CSS on the *same* problem.

These are the comparison methods for the static GA.  Everything the four
algorithms share with :class:`~optimization.ga_core.ConfigurableGA` is
literally shared code, so a difference in the results can only come from the
search strategy:

* the same representation -- a chromosome is ``[x, y, rotation]`` per machine;
* the same objective -- :class:`~feasibility.evaluator.LayoutEvaluator`, which
  returns ``+inf`` for an infeasible layout and counts every call;
* the same bound repair -- positions clipped to the machine's half-extent
  inside the workspace, rotations taken modulo 360, exactly as
  ``ConfigurableGA.apply_bounds`` does;
* the same starting point -- :func:`~optimization.ga_core.make_initial_population`
  under the run seed, so on a given (instance, seed) pair the GA and all four
  baselines begin from a bit-identical initial population;
* the same accounting -- per-iteration traces of best-so-far fitness,
  cumulative evaluations and cumulative wall-clock time, returned in the same
  :class:`~optimization.ga_core.RunResult` container the GA uses, so the
  existing analysis, statistics and figures read them without changes.

Budget
------
The GA without local search spends ``population_size * generations`` = 60 000
objective evaluations.  Every baseline is capped at the same number
(``max_evaluations``), which is the comparable budget: DE, PSO and CSS spend
one population of evaluations per iteration like the GA, while ABC spends two
(employed + onlooker), so ABC completes half as many iterations for the same
effort.  The analysis compares the methods at equal evaluations and at equal
wall-clock time as well as at the end of the run.

Parameters
----------
Every algorithm runs at its textbook default setting -- see ``BASELINES.md``
for the values, their sources, and the places where the ``+inf`` objective or
the 200-member shared population forced an explicit implementation choice.

Reproducibility
---------------
Each run owns a private ``random.Random`` and ``numpy.Generator`` seeded from
the run seed, as the GA does; no module-level RNG state is touched.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.datastruct import Machine, Point
from feasibility.evaluator import LayoutEvaluator
from optimization.ga_core import RunResult, population_diversity

#: Algorithm keys understood by :func:`build_optimizer`.
ALGORITHMS: Tuple[str, ...] = ("de", "pso", "abc", "css")

#: Display names used in labels and reports.
ALGORITHM_NAMES: Dict[str, str] = {
    "de": "Differential Evolution",
    "pso": "Particle Swarm Optimisation",
    "abc": "Artificial Bee Colony",
    "css": "Charged System Search",
}

_EPS = 1e-12


@dataclass
class MetaConfig:
    """Configuration of one baseline run.

    The population and iteration budget fields mirror :class:`GAConfig` so a
    baseline and the GA can be given the same budget by construction.  The
    per-algorithm blocks below hold each method's own parameters at their
    published default values.
    """

    algorithm: str = "de"
    label: str = ""

    # --- budget, shared with the GA ----------------------------------
    population_size: int = 200
    generations: int = 300  # iteration cap
    max_evaluations: Optional[int] = None  # default: population_size * generations
    max_seconds: Optional[float] = None

    # --- DE (Storn & Price 1997: DE/rand/1/bin, F = 0.5, CR = 0.9) ----
    de_f: float = 0.5
    de_cr: float = 0.9

    # --- PSO (Clerc & Kennedy 2002 constriction, v_max = half the range)
    pso_w: float = 0.7298
    pso_c1: float = 1.49618
    pso_c2: float = 1.49618
    pso_vmax_fraction: float = 0.5

    # --- ABC (Karaboga 2005; SN = population_size food sources) -------
    abc_limit: Optional[int] = None  # default: SN * D

    # --- CSS (Kaveh & Talatahari 2010) -------------------------------
    css_cm_fraction: float = 0.25  # charged-memory size = CP / 4
    css_ka: float = 0.5  # acceleration coefficient base
    css_kv: float = 0.5  # velocity coefficient base
    css_a_fraction: float = 0.10  # a = 0.10 * max(range), in normalised units
    css_cmcr: float = 0.95  # harmony-search boundary handling
    css_par: float = 0.10
    #: Average the electric force over the sources instead of summing it, as
    #: the paper does.  Off by default: the published sum is written for 20-40
    #: charged particles and at the 200 used here it does push about half the
    #: position genes out of the box each iteration, but averaging instead
    #: collapses the swarm onto one point and searches worse.  See
    #: BASELINES.md; the flag exists so that measurement can be repeated.
    css_mean_force: bool = False

    # --- bookkeeping --------------------------------------------------
    record_diversity: bool = True
    verbose: int = 0

    def __post_init__(self) -> None:
        if self.algorithm not in ALGORITHMS:
            raise ValueError(
                f"unknown algorithm {self.algorithm!r} (expected one of {ALGORITHMS})"
            )
        if not self.label:
            self.label = ALGORITHM_NAMES[self.algorithm]

    def evaluation_budget(self) -> int:
        """The evaluation cap, defaulting to one GA-sized budget."""
        if self.max_evaluations is not None:
            return int(self.max_evaluations)
        return int(self.population_size * self.generations)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["max_evaluations"] = self.evaluation_budget()
        return d


# ----------------------------------------------------------------------
# shared machinery
# ----------------------------------------------------------------------


class PopulationOptimizer:
    """Base class: problem, bounds, evaluation accounting and the trace loop.

    A subclass supplies ``key``, an optional :meth:`_setup` and a :meth:`_step`
    that consumes one iteration's worth of the evaluation budget.
    """

    key: str = "base"

    def __init__(
        self,
        machines: List[Machine],
        sequence: List[int],
        robot_position: Point,
        workspace_bounds: Tuple[float, float, float, float],
        config: Optional[MetaConfig] = None,
        seed: Optional[int] = None,
        instance_name: str = "instance",
    ):
        self.machines = machines
        self.sequence = list(sequence)
        self.robot_position = robot_position
        self.workspace_bounds = workspace_bounds
        self.config = config or MetaConfig(algorithm=self.key)
        self.instance_name = instance_name
        self.seed = seed

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.evaluator = LayoutEvaluator(machines, sequence, robot_position, workspace_bounds)

        # Per-gene box, identical to the one ConfigurableGA.apply_bounds
        # enforces: positions inside the workspace by the machine's half
        # extent, rotation on the circle [0, 360).
        min_x, max_x, min_y, max_y = workspace_bounds
        lower: List[float] = []
        upper: List[float] = []
        for machine in machines:
            lower.extend([min_x + machine.width / 2, min_y + machine.height / 2, 0.0])
            upper.extend([max_x - machine.width / 2, max_y - machine.height / 2, 360.0])
        self.lower = np.array(lower, dtype=float)
        self.upper = np.array(upper, dtype=float)
        self.n_genes = self.lower.size
        #: Index mask of the rotation genes (every third one).
        self.is_rotation = np.zeros(self.n_genes, dtype=bool)
        self.is_rotation[2::3] = True

        self._budget = self.config.evaluation_budget()

    # -- problem -------------------------------------------------------

    def fitness_function(self, chromosome: np.ndarray) -> float:
        return self.evaluator.fitness(chromosome)

    def decode_chromosome(self, chromosome: np.ndarray) -> List[Machine]:
        return self.evaluator.decode_to_machines(chromosome)

    def repair(self, chromosome: np.ndarray) -> np.ndarray:
        """Clip positions, wrap rotations -- ``ConfigurableGA.apply_bounds``."""
        repaired = np.asarray(chromosome, dtype=float).copy()
        positions = ~self.is_rotation
        repaired[positions] = np.clip(
            repaired[positions], self.lower[positions], self.upper[positions]
        )
        repaired[self.is_rotation] = repaired[self.is_rotation] % 360.0
        return repaired

    def random_solution(self) -> np.ndarray:
        """A uniformly random layout inside the box (ABC scouts, CSS resets)."""
        return self.lower + self.np_rng.random(self.n_genes) * (self.upper - self.lower)

    # -- budget --------------------------------------------------------

    @property
    def evaluations_used(self) -> int:
        return self.evaluator.n_evals

    def budget_exhausted(self) -> bool:
        return self.evaluations_used >= self._budget

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        return np.array([self.fitness_function(row) for row in population], dtype=float)

    # -- per-algorithm hooks -------------------------------------------

    def _setup(self, population: np.ndarray, fitness: np.ndarray) -> None:
        """Initialise algorithm state once the first population is evaluated."""

    def _step(
        self, population: np.ndarray, fitness: np.ndarray, iteration: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    # -- the loop ------------------------------------------------------

    def optimize(self, initial_population: Optional[Sequence[np.ndarray]] = None) -> RunResult:
        """Run the algorithm and return the GA's :class:`RunResult`.

        The loop has the same shape as ``ConfigurableGA.optimize``: evaluate,
        record, check the budget, then produce the next population.  Iteration
        ``k`` of a baseline is therefore recorded against exactly the same
        cumulative evaluation count as generation ``k`` of the GA (for DE, PSO
        and CSS, which like the GA spend one population per iteration).
        """
        cfg = self.config

        if initial_population is None:
            raise ValueError(
                "the baselines must be given the GA's initial population; call "
                "make_initial_population(..., seed=run_seed) and pass it in"
            )
        if len(initial_population) != cfg.population_size:
            raise ValueError(
                f"initial population size {len(initial_population)} != "
                f"population_size {cfg.population_size}"
            )
        population = np.array([np.asarray(c, dtype=float) for c in initial_population])

        n_iterations = cfg.generations
        traces = {
            key: np.full(n_iterations, np.nan)
            for key in (
                "best_so_far",
                "gen_best",
                "gen_avg",
                "diversity",
                "evaluations",
                "elapsed",
                "ls_rate",
                "mutation_rate",
                "crossover_rate",
                "feasible_fraction",
                "rl_reward",
            )
        }
        traces["rl_state"] = np.full(n_iterations, -1, dtype=int)
        traces["rl_action"] = np.full(n_iterations, -1, dtype=int)
        # No local search, no adaptive operator rates: 0 is the honest value
        # for the local-search rate, and the GA's mutation/crossover traces
        # have no counterpart here.
        traces["ls_rate"][:] = 0.0

        self.evaluator.reset_counter()
        start_time = time.perf_counter()
        start_cpu = time.process_time()

        fitness = self.evaluate_population(population)
        self._setup(population, fitness)

        best_fitness = float("inf")
        best_chromosome: Optional[np.ndarray] = None
        iterations_run = 0

        for iteration in range(n_iterations):
            elapsed = time.perf_counter() - start_time
            evals_used = self.evaluations_used

            best_idx = int(np.argmin(fitness))
            gen_best = float(fitness[best_idx])
            if gen_best < best_fitness:
                best_fitness = gen_best
                best_chromosome = population[best_idx].copy()

            finite = fitness[np.isfinite(fitness)]
            gen_avg = float(finite.mean()) if finite.size else float("nan")
            feasible_fraction = float(finite.size) / float(fitness.size)
            diversity = (
                population_diversity(population) if cfg.record_diversity else float("nan")
            )

            traces["best_so_far"][iteration] = best_fitness
            traces["gen_best"][iteration] = gen_best
            traces["gen_avg"][iteration] = gen_avg
            traces["diversity"][iteration] = diversity
            traces["evaluations"][iteration] = evals_used
            traces["elapsed"][iteration] = elapsed
            traces["feasible_fraction"][iteration] = feasible_fraction

            if cfg.verbose and iteration % cfg.verbose == 0:
                print(
                    f"[{cfg.label}] iter {iteration:>4}: best={best_fitness:.3f} "
                    f"avg={gen_avg:.3f} div={diversity:.1f} "
                    f"evals={evals_used} t={elapsed:.1f}s"
                )

            iterations_run = iteration + 1

            if self.budget_exhausted():
                break
            if cfg.max_seconds is not None and elapsed >= cfg.max_seconds:
                break

            population, fitness = self._step(population, fitness, iteration)

        execution_time = time.perf_counter() - start_time
        cpu_time = time.process_time() - start_cpu

        if best_chromosome is None:  # no finite fitness was ever seen
            best_chromosome = population[int(np.argmin(fitness))].copy()

        final_machines = self.decode_chromosome(best_chromosome)
        feasible = math.isfinite(best_fitness)
        total_distance = (
            self.evaluator.total_distance(final_machines) if feasible else float("nan")
        )

        trimmed = {key: value[:iterations_run] for key, value in traces.items()}

        return RunResult(
            label=cfg.label,
            instance_name=self.instance_name,
            seed=self.seed if self.seed is not None else -1,
            best_fitness=float(best_fitness),
            total_distance=float(total_distance),
            execution_time=float(execution_time),
            cpu_time=float(cpu_time),
            total_evaluations=int(self.evaluations_used),
            generations=int(iterations_run),
            feasible=bool(feasible),
            best_chromosome=best_chromosome,
            final_machines=final_machines,
            q_table=None,
            decoded_evolution=[],
            **trimmed,
        )


def _distinct_indices(rng: random.Random, n: int, k: int, exclude: int) -> List[int]:
    """``k`` distinct indices from ``range(n)``, none of them ``exclude``."""
    chosen: List[int] = []
    while len(chosen) < k:
        candidate = rng.randrange(n)
        if candidate != exclude and candidate not in chosen:
            chosen.append(candidate)
    return chosen


# ----------------------------------------------------------------------
# differential evolution
# ----------------------------------------------------------------------


class DifferentialEvolution(PopulationOptimizer):
    """DE/rand/1/bin (Storn & Price 1997) at its classic defaults.

    One mutant and one binomial-crossover trial per member per generation,
    accepted greedily -- so a generation costs exactly one population of
    evaluations, the same as a GA generation without local search.
    """

    key = "de"

    def _step(
        self, population: np.ndarray, fitness: np.ndarray, iteration: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        n = population.shape[0]
        new_population = population.copy()
        new_fitness = fitness.copy()

        for i in range(n):
            r1, r2, r3 = _distinct_indices(self.rng, n, 3, exclude=i)
            mutant = population[r1] + cfg.de_f * (population[r2] - population[r3])

            crossover = self.np_rng.random(self.n_genes) < cfg.de_cr
            crossover[self.rng.randrange(self.n_genes)] = True  # j_rand: at least one gene
            trial = self.repair(np.where(crossover, mutant, population[i]))

            trial_fitness = self.fitness_function(trial)
            # "<=" is Storn & Price's own acceptance rule: on the plateaus this
            # objective has (every infeasible layout scores +inf) it is what
            # lets the population drift instead of freezing.
            if trial_fitness <= fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness

            if self.budget_exhausted():
                break

        return new_population, new_fitness


# ----------------------------------------------------------------------
# particle swarm optimisation
# ----------------------------------------------------------------------


class ParticleSwarm(PopulationOptimizer):
    """Global-best PSO with the Clerc & Kennedy (2002) constriction constants.

    ``w = 0.7298``, ``c1 = c2 = 1.49618``, velocities clamped to half the range
    of each variable and initialised uniformly inside that clamp.
    """

    key = "pso"

    def _setup(self, population: np.ndarray, fitness: np.ndarray) -> None:
        span = self.upper - self.lower
        self.v_max = self.config.pso_vmax_fraction * span
        self.velocity = self.np_rng.uniform(-self.v_max, self.v_max, size=population.shape)
        self.pbest = population.copy()
        self.pbest_fitness = fitness.copy()
        best = int(np.argmin(fitness))
        self.gbest = population[best].copy()
        self.gbest_fitness = float(fitness[best])

    def _step(
        self, population: np.ndarray, fitness: np.ndarray, iteration: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        n = population.shape[0]

        r1 = self.np_rng.random(population.shape)
        r2 = self.np_rng.random(population.shape)
        self.velocity = (
            cfg.pso_w * self.velocity
            + cfg.pso_c1 * r1 * (self.pbest - population)
            + cfg.pso_c2 * r2 * (self.gbest[None, :] - population)
        )
        np.clip(self.velocity, -self.v_max, self.v_max, out=self.velocity)

        moved = population + self.velocity
        new_population = np.array([self.repair(row) for row in moved])
        new_fitness = np.empty(n, dtype=float)
        for i in range(n):
            new_fitness[i] = self.fitness_function(new_population[i])
            if self.budget_exhausted() and i + 1 < n:
                # Out of budget mid-swarm: keep the particles not yet moved.
                new_population[i + 1 :] = population[i + 1 :]
                new_fitness[i + 1 :] = fitness[i + 1 :]
                break

        improved = new_fitness < self.pbest_fitness
        self.pbest[improved] = new_population[improved]
        self.pbest_fitness[improved] = new_fitness[improved]

        best = int(np.argmin(self.pbest_fitness))
        if self.pbest_fitness[best] < self.gbest_fitness:
            self.gbest = self.pbest[best].copy()
            self.gbest_fitness = float(self.pbest_fitness[best])

        return new_population, new_fitness


# ----------------------------------------------------------------------
# artificial bee colony
# ----------------------------------------------------------------------


class ArtificialBeeColony(PopulationOptimizer):
    """ABC (Karaboga 2005) with the standard three phases.

    ``SN`` food sources are the shared initial population, so the colony is
    ``2 * SN`` bees (employed + onlooker) and one cycle costs two populations
    of evaluations: ABC completes half as many cycles as the GA, DE, PSO and
    CSS do inside the same evaluation budget.  ``limit = SN * D`` is
    Karaboga's own default abandonment threshold.

    The onlooker probabilities use the textbook fitness transform
    ``1 / (1 + f)`` for ``f >= 0`` and ``1 + |f|`` otherwise, which maps an
    infeasible layout (``f = +inf``) to probability 0 -- infeasible sources are
    never selected by onlookers, but employed bees and scouts still work on
    them, which is how they get repaired.
    """

    key = "abc"

    def _setup(self, population: np.ndarray, fitness: np.ndarray) -> None:
        self.trials = np.zeros(population.shape[0], dtype=int)
        limit = self.config.abc_limit
        self.limit = int(limit) if limit is not None else population.shape[0] * self.n_genes

    def _neighbour(self, population: np.ndarray, i: int) -> np.ndarray:
        """``v_ij = x_ij + phi * (x_ij - x_kj)`` on one random gene ``j``."""
        n = population.shape[0]
        k = self.rng.randrange(n)
        while k == i and n > 1:
            k = self.rng.randrange(n)
        j = self.rng.randrange(self.n_genes)
        phi = self.rng.uniform(-1.0, 1.0)

        candidate = population[i].copy()
        candidate[j] = population[i][j] + phi * (population[i][j] - population[k][j])
        return self.repair(candidate)

    def _try_source(self, population: np.ndarray, fitness: np.ndarray, i: int) -> None:
        candidate = self._neighbour(population, i)
        candidate_fitness = self.fitness_function(candidate)
        if candidate_fitness < fitness[i]:
            population[i] = candidate
            fitness[i] = candidate_fitness
            self.trials[i] = 0
        else:
            self.trials[i] += 1

    @staticmethod
    def _selection_fitness(fitness: np.ndarray) -> np.ndarray:
        """Karaboga's maximisation transform of a minimised objective."""
        transformed = np.where(
            fitness >= 0.0, 1.0 / (1.0 + np.maximum(fitness, 0.0)), 1.0 + np.abs(fitness)
        )
        return np.where(np.isfinite(fitness), transformed, 0.0)

    def _step(
        self, population: np.ndarray, fitness: np.ndarray, iteration: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        population = population.copy()
        fitness = fitness.copy()
        n = population.shape[0]

        # --- employed bees: one neighbour per food source ---------------
        for i in range(n):
            self._try_source(population, fitness, i)
            if self.budget_exhausted():
                return population, fitness

        # --- onlooker bees: roulette wheel over the transformed fitness --
        selection = self._selection_fitness(fitness)
        total = float(selection.sum())
        probabilities = (
            selection / total if total > 0.0 else np.full(n, 1.0 / n)
        )

        i = 0
        served = 0
        # Karaboga's roulette scans the sources in turn until SN onlookers have
        # been placed.  The pass counter is a guard, not part of the algorithm:
        # it stops the scan from spinning if a degenerate distribution makes
        # every source vanishingly unlikely, and hands any onlookers left over
        # to the best sources instead.
        # A scan places one onlooker per full sweep on average, so it needs
        # about n * n passes; the cap is far above that and only ever fires on
        # a degenerate distribution.
        passes = 0
        while served < n and passes < 20 * n * n:
            if self.rng.random() < probabilities[i]:
                served += 1
                self._try_source(population, fitness, i)
                if self.budget_exhausted():
                    return population, fitness
            i = (i + 1) % n
            passes += 1

        for target in np.argsort(fitness)[: n - served]:
            self._try_source(population, fitness, int(target))
            if self.budget_exhausted():
                return population, fitness

        # --- scout: abandon the most stagnant source --------------------
        worst = int(np.argmax(self.trials))
        if self.trials[worst] > self.limit:
            population[worst] = self.repair(self.random_solution())
            fitness[worst] = self.fitness_function(population[worst])
            self.trials[worst] = 0

        return population, fitness


# ----------------------------------------------------------------------
# charged system search
# ----------------------------------------------------------------------


class ChargedSystemSearch(PopulationOptimizer):
    """CSS (Kaveh & Talatahari 2010) with the paper's default coefficients.

    ``CP`` charged particles (the shared initial population), a charged memory
    of ``CP / 4`` best-so-far solutions that also acts as a source of forces,
    ``k_a = 0.5 (1 + iter / iter_max)``, ``k_v = 0.5 (1 - iter / iter_max)``
    and ``dt = 1``.  A particle that leaves the search space is corrected by
    the paper's harmony-search rule (``CMCR = 0.95``, ``PAR = 0.10``) before
    the box repair.

    Two things the paper leaves to the implementation had to be pinned down
    here:

    * The force law is evaluated in the *normalised* search space -- every
      variable scaled to [0, 1] by its own range -- so the radius
      ``a = 0.10 * max(range)`` takes the paper's value of 0.10.  In raw units
      the 360-degree rotation gene would set ``a = 36`` and the ``1 / a**3``
      near-field term would freeze the swarm.
    * ``+inf`` fitness cannot be normalised into a charge, so for the charge
      and moving-probability formulas only, an infeasible particle is ranked
      one fitness-range beyond the worst feasible one.  The objective itself
      is untouched.
    """

    key = "css"

    def _setup(self, population: np.ndarray, fitness: np.ndarray) -> None:
        cfg = self.config
        n = population.shape[0]
        self.cm_size = max(1, int(round(cfg.css_cm_fraction * n)))
        # Velocities live in the normalised space the forces are computed in.
        self.velocity = np.zeros_like(population)
        self.span = np.where(self.upper - self.lower > _EPS, self.upper - self.lower, 1.0)
        self.a = cfg.css_a_fraction  # max range is 1 once normalised

        order = np.argsort(fitness)[: self.cm_size]
        self.cm = population[order].copy()
        self.cm_fitness = fitness[order].copy()

    def _to_unit(self, x: np.ndarray) -> np.ndarray:
        """Map a layout (or a stack of them) onto the unit hypercube."""
        return (np.asarray(x, dtype=float) - self.lower) / self.span

    def _from_unit(self, z: np.ndarray) -> np.ndarray:
        return self.lower + np.asarray(z, dtype=float) * self.span

    @staticmethod
    def _rankable(fitness: np.ndarray) -> np.ndarray:
        """Finite stand-ins for ``+inf``, used only by the force formulas."""
        finite = fitness[np.isfinite(fitness)]
        if finite.size == 0:
            return np.zeros_like(fitness)
        worst = float(finite.max())
        spread = float(finite.max() - finite.min()) or 1.0
        return np.where(np.isfinite(fitness), fitness, worst + spread)

    def _update_memory(self, population: np.ndarray, fitness: np.ndarray) -> None:
        """Keep the ``cm_size`` best *distinct* solutions seen so far.

        Distinct matters: a converging swarm re-discovers its own best layout
        many times, and a memory holding N copies of one point would attract
        every particle to that single location.
        """
        pool = np.vstack([self.cm, population])
        pool_fitness = np.concatenate([self.cm_fitness, fitness])

        keep: List[int] = []
        for index in np.argsort(pool_fitness):
            if any(np.allclose(pool[index], pool[other]) for other in keep):
                continue
            keep.append(int(index))
            if len(keep) == self.cm_size:
                break
        self.cm = pool[keep].copy()
        self.cm_fitness = pool_fitness[keep].copy()

    def _harmony_repair(self, candidate: np.ndarray) -> np.ndarray:
        """Harmony-search correction of out-of-range genes (Kaveh's rule)."""
        out_of_range = (candidate < self.lower) | (candidate > self.upper)
        out_of_range &= ~self.is_rotation  # rotation is periodic, never "outside"
        if not out_of_range.any():
            return self.repair(candidate)

        corrected = candidate.copy()
        span = self.upper - self.lower
        for j in np.flatnonzero(out_of_range):
            if self.rng.random() < self.config.css_cmcr:
                source = self.cm[self.rng.randrange(self.cm.shape[0])]
                value = source[j]
                if self.rng.random() < self.config.css_par:
                    value += self.rng.uniform(-1.0, 1.0) * 0.01 * span[j]
            else:
                value = self.rng.uniform(self.lower[j], self.upper[j])
            corrected[j] = value
        return self.repair(corrected)

    def _step(
        self, population: np.ndarray, fitness: np.ndarray, iteration: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        n = population.shape[0]

        # Sources of the electric force: the particles plus the charged memory,
        # all in the normalised space the force law is defined on.
        sources = self._to_unit(np.vstack([population, self.cm]))
        source_fitness = np.concatenate([fitness, self.cm_fitness])
        ranked_sources = self._rankable(source_fitness)
        ranked_particles = ranked_sources[:n]

        best = float(ranked_sources.min())
        worst = float(ranked_sources.max())
        charge = (
            np.ones_like(ranked_sources)
            if worst - best < _EPS
            else (ranked_sources - worst) / (best - worst)
        )
        best_position = sources[int(np.argmin(ranked_sources))]

        progress = (iteration + 1) / max(cfg.generations, 1)
        ka = cfg.css_ka * (1.0 + progress)
        kv = cfg.css_kv * (1.0 - progress)

        new_population = np.empty_like(population)
        new_fitness = np.empty(n, dtype=float)
        stopped_at = n

        for j in range(n):
            zj = sources[j]
            # Separation distance, normalised by the distance of the pair's
            # midpoint from the best solution (Kaveh & Talatahari, eq. 6).
            delta = sources - zj[None, :]
            distance = np.linalg.norm(delta, axis=1)
            midpoint = 0.5 * (sources + zj[None, :]) - best_position[None, :]
            separation = distance / (np.linalg.norm(midpoint, axis=1) + _EPS)

            near = separation < self.a
            kernel = np.where(
                near,
                charge * separation / (self.a ** 3),
                charge / np.maximum(separation, _EPS) ** 2,
            )

            # Moving probability (eq. 9): a worse particle is always attracted
            # to a better one; a better one follows a worse one only sometimes.
            gap = ranked_sources - best
            denominator = ranked_particles[j] - ranked_sources
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(np.abs(denominator) < _EPS, np.inf, gap / denominator)
            probability = (
                (ratio > self.np_rng.random(sources.shape[0]))
                | (ranked_sources < ranked_particles[j])
            ).astype(float)

            weight = kernel * probability
            weight[j] = 0.0  # a particle exerts no force on itself
            force = weight @ delta
            if cfg.css_mean_force:
                force /= sources.shape[0]

            r1 = self.np_rng.random(self.n_genes)
            r2 = self.np_rng.random(self.n_genes)
            # dt = 1 and m_j = q_j, so F_j / m_j is the summation above.
            moved = zj + r1 * ka * force + r2 * kv * self.velocity[j]
            candidate = self._harmony_repair(self._from_unit(moved))

            self.velocity[j] = self._to_unit(candidate) - zj
            new_population[j] = candidate
            new_fitness[j] = self.fitness_function(candidate)

            if self.budget_exhausted() and j + 1 < n:
                stopped_at = j + 1
                new_population[stopped_at:] = population[stopped_at:]
                new_fitness[stopped_at:] = fitness[stopped_at:]
                break

        # Step 7: new particles better than the worst memory entries enter the
        # charged memory.  The traffic is one-way - the memory records good
        # layouts and pulls the particles towards them through the force term
        # above, it does not overwrite them.
        self._update_memory(new_population[:stopped_at], new_fitness[:stopped_at])

        return new_population, new_fitness


# ----------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------

_REGISTRY: Dict[str, type] = {
    cls.key: cls
    for cls in (DifferentialEvolution, ParticleSwarm, ArtificialBeeColony, ChargedSystemSearch)
}


def build_optimizer(
    algorithm: str,
    machines: List[Machine],
    sequence: List[int],
    robot_position: Point,
    workspace_bounds: Tuple[float, float, float, float],
    config: Optional[MetaConfig] = None,
    seed: Optional[int] = None,
    instance_name: str = "instance",
) -> PopulationOptimizer:
    """Instantiate one baseline by key (``de`` / ``pso`` / ``abc`` / ``css``)."""
    try:
        cls = _REGISTRY[algorithm]
    except KeyError:
        raise ValueError(
            f"unknown algorithm {algorithm!r} (expected one of {ALGORITHMS})"
        ) from None
    config = config or MetaConfig(algorithm=algorithm)
    if config.algorithm != algorithm:
        raise ValueError(
            f"configuration is for {config.algorithm!r} but {algorithm!r} was requested"
        )
    return cls(
        machines,
        sequence,
        robot_position,
        workspace_bounds,
        config=config,
        seed=seed,
        instance_name=instance_name,
    )
