"""
One genetic algorithm, many controllers.

``ConfigurableGA`` is a single GA implementation where both where local
search is applied and how its rate (plus mutation/crossover rates) is
controlled are explicit configuration options:

* ``ls_target``      - ``"offspring"``, ``"elites"`` or ``"none"``
* ``ls_controller``  - ``"static"`` (fixed rate), ``"schedule"`` (deterministic
  generation-dependent rate) or ``"rl"`` (Q-learning agent)

Every other component - representation, decoder, fitness, selection,
crossover, mutation operator, elitism, local-search neighbourhood and step
count - is shared code, so two configurations differ only in the dimension
being studied.

Budget accounting
--------------------------------
Every objective-function evaluation is counted, *including* the ones consumed
inside local search, and every generation records the cumulative evaluation
count and the cumulative wall-clock time.  The resulting traces let the
analysis compare methods at an equal number of evaluations and at equal
wall-clock time instead of at an equal number of generations.

Reproducibility
--------------------------
Each run owns a private ``random.Random`` and ``numpy.Generator`` seeded from
the run seed; no module-level RNG state is touched, so runs are reproducible
and safe to execute in parallel processes.
"""

from __future__ import annotations

import copy
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from optimization.rl_control import BASE_ACTIONS, QLearningAgent, StateBins
from core.datastruct import Machine, Point
from feasibility.evaluator import LayoutEvaluator

#: Deterministic local-search schedule: 10% of the
#: generations budget at 0.1, the next third at 0.5, the last third at 1.0.
DEFAULT_LS_SCHEDULE: Tuple[Tuple[int, float], ...] = ((0, 0.1), (100, 0.5), (200, 1.0))

#: Reward handed to the agent for a generation that failed to improve the best
#: fitness.  The value the submitted study used is -0.1.
DEFAULT_RL_STALL_PENALTY: float = -0.1


def _default_stall_penalty() -> float:
    raw = os.environ.get("FYP_RL_STALL_PENALTY")
    if raw is None:
        return DEFAULT_RL_STALL_PENALTY
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"FYP_RL_STALL_PENALTY must be a number, got {raw!r}"
        ) from exc
    if not math.isfinite(value) or value > 0.0:
        raise ValueError(
            f"FYP_RL_STALL_PENALTY must be a finite value <= 0 (it is a "
            f"penalty, e.g. -0.01), got {value}"
        )
    return value


@dataclass
class GAConfig:
    """Everything that defines an algorithm variant."""

    label: str = "GA"

    ls_target: str = "offspring"  # offspring | elites | none
    ls_controller: str = "static"  # static | schedule | rl
    ls_rate: float = 0.1  # used when ls_controller == "static"; also the RL start value
    ls_schedule: Tuple[Tuple[int, float], ...] = DEFAULT_LS_SCHEDULE

    # --- shared GA settings (identical across variants) --------------
    population_size: int = 200
    generations: int = 300
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elitism_rate: float = 0.1
    tournament_k: int = 3
    ls_iterations: int = 5

    # --- RL settings -------------------------------------------------
    control_interval: int = 1
    agent_key: Optional[str] = None
    allowed_actions: Optional[Tuple[str, ...]] = None
    rl_cost_penalty: float = 0.0 
    rl_stall_penalty: float = field(default_factory=_default_stall_penalty)
    rl_learn_online: bool = True  # keep updating Q during evaluation runs
    mutation_rate_bounds: Tuple[float, float, float] = (0.01, 0.8, 0.02)  # min, max, step
    crossover_rate_bounds: Tuple[float, float, float] = (0.2, 1.0, 0.05)
    ls_rate_step: float = 0.1

    # --- budget caps --
    max_evaluations: Optional[int] = None
    max_seconds: Optional[float] = None

    # --- bookkeeping -------------------------------------------------
    snapshot_interval: int = 0  # >0 stores decoded layouts every k generations
    record_diversity: bool = True
    verbose: int = 0

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["ls_schedule"] = [list(x) for x in self.ls_schedule]
        d["allowed_actions"] = list(self.allowed_actions) if self.allowed_actions else None
        return d

    def action_set(self) -> Tuple[str, ...]:
        """The ordered action names this configuration exposes to the agent."""
        if self.allowed_actions is None:
            return BASE_ACTIONS
        unknown = set(self.allowed_actions) - set(BASE_ACTIONS)
        if unknown:
            raise ValueError(f"unknown action(s) in allowed_actions: {sorted(unknown)}")
        return tuple(name for name in BASE_ACTIONS if name in self.allowed_actions)


@dataclass
class RunResult:
    """Outcome of one GA run, including the anytime traces."""

    label: str
    instance_name: str
    seed: int

    best_fitness: float
    total_distance: float
    execution_time: float
    cpu_time: float
    total_evaluations: int
    generations: int
    feasible: bool

    # per-generation traces (length == generations)
    best_so_far: np.ndarray
    gen_best: np.ndarray
    gen_avg: np.ndarray
    diversity: np.ndarray
    evaluations: np.ndarray
    elapsed: np.ndarray
    ls_rate: np.ndarray
    mutation_rate: np.ndarray
    crossover_rate: np.ndarray
    feasible_fraction: np.ndarray

    # RL traces (all -1 / 0 for non-RL variants)
    rl_state: np.ndarray
    rl_action: np.ndarray
    rl_reward: np.ndarray

    best_chromosome: np.ndarray
    final_machines: List[Machine] = field(default_factory=list, repr=False)
    q_table: Optional[np.ndarray] = None
    decoded_evolution: List[List[Machine]] = field(default_factory=list, repr=False)

    def trace_dict(self) -> Dict[str, np.ndarray]:
        return {
            "best_so_far": self.best_so_far,
            "gen_best": self.gen_best,
            "gen_avg": self.gen_avg,
            "diversity": self.diversity,
            "evaluations": self.evaluations,
            "elapsed": self.elapsed,
            "ls_rate": self.ls_rate,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "feasible_fraction": self.feasible_fraction,
            "rl_state": self.rl_state,
            "rl_action": self.rl_action,
            "rl_reward": self.rl_reward,
            "best_chromosome": self.best_chromosome,
        }


class ConfigurableGA:
    """The shared GA. Behaviour is fully determined by :class:`GAConfig`."""

    def __init__(
        self,
        machines: List[Machine],
        sequence: List[int],
        robot_position: Point,
        workspace_bounds: Tuple[float, float, float, float],
        config: Optional[GAConfig] = None,
        rl_agent: Optional[QLearningAgent] = None,
        state_bins: Optional[StateBins] = None,
        seed: Optional[int] = None,
        instance_name: str = "instance",
    ):
        self.machines = machines
        self.sequence = list(sequence)
        self.robot_position = robot_position
        self.workspace_bounds = workspace_bounds
        self.config = config or GAConfig()
        self.instance_name = instance_name
        self.seed = seed

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.evaluator = LayoutEvaluator(machines, sequence, robot_position, workspace_bounds)

        # Live parameters
        self.mutation_rate = self.config.mutation_rate
        self.crossover_rate = self.config.crossover_rate
        self.elitism_rate = self.config.elitism_rate
        self.local_search_rate = self.config.ls_rate

        if self.config.ls_controller == "rl":
            # The RL state is a function of population diversity, so the
            # diversity measurement is not optional for this controller.
            self.config.record_diversity = True
            self.state_bins = state_bins or StateBins()
            self.action_names = self.config.action_set()
            self.rl_agent = rl_agent or QLearningAgent(
                n_states=self.state_bins.n_states,
                n_actions=len(self.action_names),
                seed=seed,
            )
            if self.rl_agent.n_actions != len(self.action_names):
                raise ValueError(
                    f"agent has {self.rl_agent.n_actions} actions but the configuration "
                    f"expects {len(self.action_names)} ({self.action_names})"
                )
        else:
            self.state_bins = state_bins
            self.rl_agent = None
            self.action_names = ()

    # ------------------------------------------------------------------
    # fitness / decoding
    # ------------------------------------------------------------------

    def fitness_function(self, chromosome: np.ndarray) -> float:
        return self.evaluator.fitness(chromosome)

    def decode_chromosome(self, chromosome: np.ndarray) -> List[Machine]:
        return self.evaluator.decode_to_machines(chromosome)

    # ------------------------------------------------------------------
    # population initialisation
    # ------------------------------------------------------------------

    def create_initial_population(self) -> List[np.ndarray]:
        """Quadrant-seeded random layouts (unchanged from the original GA)."""
        return make_initial_population(
            self.machines,
            self.workspace_bounds,
            self.config.population_size,
            rng=self.rng,
        )

    # ------------------------------------------------------------------
    # variation operators
    # ------------------------------------------------------------------

    def apply_bounds(self, chromosome: np.ndarray) -> np.ndarray:
        min_x, max_x, min_y, max_y = self.workspace_bounds
        for i in range(0, len(chromosome), 3):
            machine = self.machines[i // 3]
            chromosome[i] = np.clip(
                chromosome[i], min_x + machine.width / 2, max_x - machine.width / 2
            )
            chromosome[i + 1] = np.clip(
                chromosome[i + 1], min_y + machine.height / 2, max_y - machine.height / 2
            )
            chromosome[i + 2] = chromosome[i + 2] % 360
        return chromosome

    def tournament_selection(
        self, population: Sequence[np.ndarray], fitness_scores: Sequence[float]
    ) -> np.ndarray:
        k = self.config.tournament_k
        indices = self.rng.sample(range(len(population)), k)
        winner = min(indices, key=lambda i: fitness_scores[i])
        return population[winner].copy()

    def crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover."""
        if self.rng.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        child1, child2 = parent1.copy(), parent2.copy()
        swap = self.np_rng.random(len(parent1)) < 0.5
        child1[swap], child2[swap] = parent2[swap], parent1[swap]
        return child1, child2

    def gaussian_mutation(self, chromosome: np.ndarray) -> np.ndarray:
        """Per-machine Gaussian jitter with a random 90-degree re-orientation."""
        mutated = chromosome.copy()
        min_x, max_x, min_y, max_y = self.workspace_bounds
        rot_options = (0.0, 90.0, 180.0, 270.0)

        for i in range(0, len(chromosome), 3):
            if self.rng.random() < self.mutation_rate:
                mutated[i] += self.rng.gauss(0, 2.0)
                mutated[i + 1] += self.rng.gauss(0, 2.0)
                mutated[i + 2] = self.rng.choice(rot_options)

                machine = self.machines[i // 3]
                mutated[i] = np.clip(
                    mutated[i], min_x + machine.width / 2, max_x - machine.width / 2
                )
                mutated[i + 1] = np.clip(
                    mutated[i + 1], min_y + machine.height / 2, max_y - machine.height / 2
                )
        return self.apply_bounds(mutated)

    def local_search(self, chromosome: np.ndarray, iterations: Optional[int] = None) -> np.ndarray:
        """First-improvement hill climbing on a single randomly chosen gene.

        Costs ``iterations + 1`` objective-function evaluations, all of which
        are counted towards the budget.
        """
        iterations = self.config.ls_iterations if iterations is None else iterations
        current = chromosome.copy()
        current_fitness = self.fitness_function(current)

        for _ in range(iterations):
            neighbour = current.copy()
            idx = self.rng.randrange(len(neighbour))
            if idx % 3 == 2:  # rotation gene
                neighbour[idx] = (neighbour[idx] + self.rng.gauss(0, 5.0)) % 360
            else:  # position gene
                neighbour[idx] += self.rng.gauss(0, 0.5)

            neighbour_fitness = self.fitness_function(neighbour)
            if neighbour_fitness < current_fitness:
                current = neighbour
                current_fitness = neighbour_fitness

        return current

    # ------------------------------------------------------------------
    # diversity
    # ------------------------------------------------------------------

    def calculate_diversity(self, population: Sequence[np.ndarray]) -> float:
        """Mean pairwise Euclidean distance between chromosomes (vectorised)."""
        return population_diversity(population)

    # ------------------------------------------------------------------
    # local-search rate controllers
    # ------------------------------------------------------------------

    def scheduled_ls_rate(self, generation: int) -> float:
        """Deterministic schedule baseline."""
        rate = self.config.ls_schedule[0][1]
        for start_gen, value in self.config.ls_schedule:
            if generation >= start_gen:
                rate = value
        return rate

    def _adjust(self, name: str, direction: int) -> None:
        if name == "mutation_rate":
            lo, hi, step = self.config.mutation_rate_bounds
            self.mutation_rate = min(hi, max(lo, self.mutation_rate + direction * step))
        elif name == "crossover_rate":
            lo, hi, step = self.config.crossover_rate_bounds
            self.crossover_rate = min(hi, max(lo, self.crossover_rate + direction * step))
        elif name == "local_search_rate":
            step = self.config.ls_rate_step
            self.local_search_rate = min(1.0, max(0.0, self.local_search_rate + direction * step))

    def apply_action(self, action: int) -> None:
        """Apply the RL action named by ``self.action_names[action]``."""
        name = self.action_names[action]

        if name == "mutation_rate_up":
            self._adjust("mutation_rate", +1)
        elif name == "mutation_rate_down":
            self._adjust("mutation_rate", -1)
        elif name == "crossover_rate_up":
            self._adjust("crossover_rate", +1)
        elif name == "crossover_rate_down":
            self._adjust("crossover_rate", -1)
        elif name == "local_search_rate_up":
            self._adjust("local_search_rate", +1)
        elif name == "local_search_rate_down":
            self._adjust("local_search_rate", -1)
        elif name == "soft_reset":
            self.mutation_rate = 0.5 * self.mutation_rate + 0.5 * self.config.mutation_rate
            self.crossover_rate = 0.5 * self.crossover_rate + 0.5 * self.config.crossover_rate
            self.local_search_rate = 0.5 * self.local_search_rate
        else:  # pragma: no cover - guards against a typo in the action table
            raise ValueError(f"unknown action {name!r}")

    def compute_reward(
        self, prev_best: float, current_best: float, extra_evaluations: int
    ) -> float:
        """Relative best-fitness improvement, optionally penalised by effort.

        A generation that did not improve the best fitness scores
        ``rl_stall_penalty`` (-0.1 unless configured otherwise), which is what
        sets the scale of "stalling is bad" against the relative improvements
        """
        if not (math.isfinite(prev_best) and math.isfinite(current_best)):
            # The whole population is still infeasible; a relative improvement
            # would be inf/inf = NaN and would poison the Q-table.
            reward = 0.0
        elif prev_best > current_best:
            reward = (prev_best - current_best) / abs(prev_best)
        else:
            reward = self.config.rl_stall_penalty

        if self.config.rl_cost_penalty:
            cost = extra_evaluations / max(self.config.population_size, 1)
            reward -= self.config.rl_cost_penalty * cost
        return reward

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def optimize(self, initial_population: Optional[Sequence[np.ndarray]] = None) -> RunResult:
        cfg = self.config

        if initial_population is None:
            population = self.create_initial_population()
        else:
            if len(initial_population) != cfg.population_size:
                raise ValueError(
                    f"initial population size {len(initial_population)} != "
                    f"population_size {cfg.population_size}"
                )
            population = [chromosome.copy() for chromosome in initial_population]

        n_generations = cfg.generations
        traces = {
            key: np.full(n_generations, np.nan)
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
        traces["rl_state"] = np.full(n_generations, -1, dtype=int)
        traces["rl_action"] = np.full(n_generations, -1, dtype=int)

        best_fitness = float("inf")
        best_chromosome: Optional[np.ndarray] = None
        decoded_evolution: List[List[Machine]] = []

        prev_state: Optional[int] = None
        prev_action: Optional[int] = None
        prev_best = float("inf")
        evals_at_prev_gen = 0

        self.evaluator.reset_counter()
        start_time = time.perf_counter()
        # CPU time is recorded alongside wall-clock time: it is far less
        # sensitive to other processes on the machine, which matters when runs
        # are executed in parallel.
        start_cpu = time.process_time()
        generations_run = 0

        for generation in range(n_generations):
            # ---- evaluate ------------------------------------------------
            fitness_scores = [self.fitness_function(chromosome) for chromosome in population]
            evals_used = self.evaluator.n_evals
            elapsed = time.perf_counter() - start_time

            best_idx = int(np.argmin(fitness_scores))
            gen_best = fitness_scores[best_idx]
            if gen_best < best_fitness:
                best_fitness = gen_best
                best_chromosome = population[best_idx].copy()

            finite = [f for f in fitness_scores if math.isfinite(f)]
            gen_avg = float(np.mean(finite)) if finite else float("nan")
            feasible_fraction = len(finite) / len(fitness_scores)

            diversity = (
                self.calculate_diversity(population) if cfg.record_diversity else float("nan")
            )

            if cfg.snapshot_interval and generation % cfg.snapshot_interval == 0:
                decoded_evolution.append(
                    copy.deepcopy(self.decode_chromosome(population[best_idx]))
                )

            # ---- controller ---------------------------------------------
            state = -1
            action = -1
            reward = 0.0

            if cfg.ls_controller == "static":
                self.local_search_rate = cfg.ls_rate
            elif cfg.ls_controller == "schedule":
                self.local_search_rate = self.scheduled_ls_rate(generation)
            elif cfg.ls_controller == "rl":
                relative_improvement = (
                    (prev_best - gen_best) / abs(prev_best)
                    if math.isfinite(prev_best) and prev_best != 0
                    else 0.0
                )
                state = self.state_bins.state(diversity, relative_improvement)

                if generation > 0 and prev_action is not None and prev_action >= 0:
                    reward = self.compute_reward(
                        prev_best, gen_best, evals_used - evals_at_prev_gen - cfg.population_size
                    )
                    if cfg.rl_learn_online:
                        self.rl_agent.update(prev_state, prev_action, reward, state)

                if generation % cfg.control_interval == 0:
                    action = int(self.rl_agent.get_action(state))
                    self.apply_action(action)

            # ---- record --------------------------------------------------
            traces["best_so_far"][generation] = best_fitness
            traces["gen_best"][generation] = gen_best
            traces["gen_avg"][generation] = gen_avg
            traces["diversity"][generation] = diversity
            traces["evaluations"][generation] = evals_used
            traces["elapsed"][generation] = elapsed
            traces["ls_rate"][generation] = self.local_search_rate
            traces["mutation_rate"][generation] = self.mutation_rate
            traces["crossover_rate"][generation] = self.crossover_rate
            traces["feasible_fraction"][generation] = feasible_fraction
            traces["rl_state"][generation] = state
            traces["rl_action"][generation] = action
            traces["rl_reward"][generation] = reward

            if cfg.verbose and generation % cfg.verbose == 0:
                print(
                    f"[{cfg.label}] gen {generation:>4}: best={best_fitness:.3f} "
                    f"avg={gen_avg:.3f} div={diversity:.1f} ls={self.local_search_rate:.2f} "
                    f"evals={evals_used} t={elapsed:.1f}s"
                )

            prev_state = state
            prev_action = action
            prev_best = gen_best
            evals_at_prev_gen = evals_used
            generations_run = generation + 1

            # ---- stop early if a budget cap was requested ----------------
            if cfg.max_evaluations is not None and evals_used >= cfg.max_evaluations:
                break
            if cfg.max_seconds is not None and elapsed >= cfg.max_seconds:
                break

            # ---- reproduce -----------------------------------------------
            population = self._next_generation(population, fitness_scores)

        execution_time = time.perf_counter() - start_time
        cpu_time = time.process_time() - start_cpu

        if best_chromosome is None:
            # No feasible layout was ever found: report the last incumbent so
            # the run still yields a decodable layout.
            best_chromosome = population[int(np.argmin(fitness_scores))].copy()

        final_machines = self.decode_chromosome(best_chromosome)
        feasible = math.isfinite(best_fitness)
        total_distance = (
            self.evaluator.total_distance(final_machines) if feasible else float("nan")
        )

        trimmed = {key: value[:generations_run] for key, value in traces.items()}

        return RunResult(
            label=cfg.label,
            instance_name=self.instance_name,
            seed=self.seed if self.seed is not None else -1,
            best_fitness=float(best_fitness),
            total_distance=float(total_distance),
            execution_time=float(execution_time),
            cpu_time=float(cpu_time),
            total_evaluations=int(self.evaluator.n_evals),
            generations=int(generations_run),
            feasible=bool(feasible),
            best_chromosome=best_chromosome,
            final_machines=final_machines,
            q_table=self.rl_agent.q_table.copy() if self.rl_agent is not None else None,
            decoded_evolution=decoded_evolution,
            **trimmed,
        )

    # ------------------------------------------------------------------

    def _next_generation(
        self, population: List[np.ndarray], fitness_scores: List[float]
    ) -> List[np.ndarray]:
        """Elitism + tournament selection + crossover + mutation + local search.

        The *only* thing that ``ls_target`` changes is which individuals the
        local-search operator is offered to.
        """
        cfg = self.config
        new_population: List[np.ndarray] = []

        elite_count = max(1, int(self.elitism_rate * cfg.population_size))
        elite_indices = np.argsort(fitness_scores)[:elite_count]
        for idx in elite_indices:
            elite = population[idx].copy()
            if cfg.ls_target == "elites" and self.rng.random() < self.local_search_rate:
                elite = self.local_search(elite)
            new_population.append(elite)

        while len(new_population) < cfg.population_size:
            parent1 = self.tournament_selection(population, fitness_scores)
            parent2 = self.tournament_selection(population, fitness_scores)
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.gaussian_mutation(child1)
            child2 = self.gaussian_mutation(child2)

            if cfg.ls_target == "offspring" and self.rng.random() < self.local_search_rate:
                child1 = self.local_search(child1)

            new_population.extend([child1, child2])

        return new_population[: cfg.population_size]


# ----------------------------------------------------------------------
# shared initial populations
# ----------------------------------------------------------------------


def population_diversity(population: Sequence[np.ndarray]) -> float:
    """Mean pairwise Euclidean distance between the members of a population.

    Shared by every optimiser in this project (the GA and the
    :mod:`optimization.metaheuristics` baselines) so the diversity trace means
    the same thing whichever algorithm produced it.
    """
    if len(population) < 2:
        return 1.0
    matrix = np.asarray(population, dtype=float)
    diff = matrix[:, None, :] - matrix[None, :, :]
    distances = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    n = matrix.shape[0]
    return float(distances.sum() / (n * (n - 1)))


def make_initial_population(
    machines: List[Machine],
    workspace_bounds: Tuple[float, float, float, float],
    population_size: int,
    rng: Optional[random.Random] = None,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """Quadrant-seeded random layouts.

    All variants compared on the same (instance, seed) pair are started from
    the *same* initial population, which is what makes per-seed paired
    statistics between variants meaningful.
    """
    rng = rng or random.Random(seed)
    min_x, max_x, min_y, max_y = workspace_bounds
    quadrants = [
        (min_x, 0, min_y, 0),
        (0, max_x, min_y, 0),
        (min_x, 0, 0, max_y),
        (0, max_x, 0, max_y),
    ]

    population: List[np.ndarray] = []
    for _ in range(population_size):
        genes: List[float] = []
        for i, machine in enumerate(machines):
            if i < len(quadrants):
                qx1, qx2, qy1, qy2 = quadrants[i]
                x = rng.uniform(qx1 + machine.width / 2, qx2 - machine.width / 2)
                y = rng.uniform(qy1 + machine.height / 2, qy2 - machine.height / 2)
            else:
                x = rng.uniform(min_x + machine.width / 2, max_x - machine.width / 2)
                y = rng.uniform(min_y + machine.height / 2, max_y - machine.height / 2)
            genes.extend([x, y, rng.uniform(0, 360)])
        population.append(np.array(genes))
    return population
