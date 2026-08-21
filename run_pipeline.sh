#!/usr/bin/env bash
#
# One-command runner for the full study.
#
#   ./run_pipeline.sh              # rebuild everything, overwriting results/
#   ./run_pipeline.sh --replicate  # reuse the shipped agents, main grid only
#   ./run_pipeline.sh --help       # all options
#
# Stages: instances -> calibrate -> train -> main -> timing -> analyze
#         -> figures.
#
# Defaults match the settings recorded in results/runs_metadata.json:
# control interval 1, 15 training instances, 10 test instances, 10 seeds per
# instance, 300 generations, population 200, agent seeds 0 1 2.
#
set -euo pipefail

# ----------------------------------------------------------------------
# defaults - the archived study's protocol
# ----------------------------------------------------------------------
INTERVAL=1             # generations between RL control decisions
TRAIN_PROBLEMS=15      # instances in the training split
TEST_PROBLEMS=10       # instances in the test split
RUNS=10                # runs per problem (training episodes and test seeds)
TIMING_RUNS=3          # seeds per problem in the sequential wall-clock pass
GENERATIONS=300
POPULATION=200
AGENT_SEEDS="0 1 2"
RL_PENALTY=""          # empty = GAConfig's default (-0.1), what the archived study used
WORKERS=0              # 0 = auto
STUDY=""               # empty = write into results/ (overwrites the archived study)
PYTHON_BIN="${PYTHON:-}"
ABLATION_AGENTS=0      # 1 = also train the point-4b per-parameter ablation agents
REPLICATE=0            # 1 = reuse the shipped agents, run main/analyze/figures
DRY_RUN=0
FROM_STAGE="instances"

usage() {
  awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "$0"
  cat <<EOF

Options
  --replicate         reuse the shipped agents in results/agents and run only
                      main + analyze + figures.  This is the true replication
                      of the archived numbers; everything else retrains.
  --study NAME        write to results/NAME and train into results/agents_NAME
                      instead of overwriting the archived results/
  --interval N        control interval in generations        (default $INTERVAL)
  --train-problems N  instances in the training split        (default $TRAIN_PROBLEMS)
  --test-problems N   instances in the test split            (default $TEST_PROBLEMS)
  --runs N            runs per problem                       (default $RUNS)
  --timing-runs N     seeds for the sequential timing pass   (default $TIMING_RUNS)
  --generations N     GA generations per run                 (default $GENERATIONS)
  --population N      GA population size                     (default $POPULATION)
  --agent-seeds "..." independent training repetitions       (default "$AGENT_SEEDS")
  --rl-penalty X      reward for a non-improving generation  (default -0.1)
                      must be <= 0; applies to training and evaluation
  --workers N         parallel workers                       (default auto)
  --python PATH       Python interpreter (default: auto-detect; or set \$PYTHON)
  --train-ablation-agents  also train the point-4b per-parameter ablation agents
                      (lsonly/mutonly/xoveronly/noLS); off by default since
                      nothing in main/timing/analyze/figures evaluates them yet
  --from STAGE        instances|calibrate|train|main|timing|analyze
  --dry-run           print the commands without running them
  -h, --help          this message

Outputs
  results/<study>/    runs.csv, raw traces, analysis/, figures/
  results/agents*/    trained agents, state bins
  logs/<study>_<stamp>.log

WARNING  Without --study, every stage opens its outputs in "w" mode and will
         overwrite the archived results/ in place.  There is no
         backup and no prompt.

Examples
  ./run_pipeline.sh --replicate
  ./run_pipeline.sh --study rerun
  ./run_pipeline.sh --study quick --runs 3 --generations 50
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --replicate)          REPLICATE=1; shift ;;
    --interval)           INTERVAL="$2"; shift 2 ;;
    --train-problems)     TRAIN_PROBLEMS="$2"; shift 2 ;;
    --test-problems)      TEST_PROBLEMS="$2"; shift 2 ;;
    --runs)               RUNS="$2"; shift 2 ;;
    --timing-runs)        TIMING_RUNS="$2"; shift 2 ;;
    --generations)        GENERATIONS="$2"; shift 2 ;;
    --population)         POPULATION="$2"; shift 2 ;;
    --agent-seeds)        AGENT_SEEDS="$2"; shift 2 ;;
    --rl-penalty)         RL_PENALTY="$2"; shift 2 ;;
    --workers)            WORKERS="$2"; shift 2 ;;
    --study)              STUDY="$2"; shift 2 ;;
    --python)             PYTHON_BIN="$2"; shift 2 ;;
    --train-ablation-agents) ABLATION_AGENTS=1; shift ;;
    --from)               FROM_STAGE="$2"; shift 2 ;;
    --dry-run)            DRY_RUN=1; shift ;;
    -h|--help)            usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"              # this directory, so "python -m <module>" resolves

# ----------------------------------------------------------------------
# locate a usable Python (shared helper; explains itself on failure)
# ----------------------------------------------------------------------
# shellcheck source=_detect_python.sh
. "$HERE/_detect_python.sh"
require_python || exit 1

if [ -n "$STUDY" ]; then
  RESULTS_DIR="results/${STUDY}"
  AGENT_DIR="results/agents_${STUDY}"
else
  STUDY="main"
  RESULTS_DIR="results"
  AGENT_DIR="results/agents"
fi
BINS="${AGENT_DIR}/state_bins.json"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/${STUDY}_${STAMP}.log"
mkdir -p logs "$AGENT_DIR"

# --replicate reuses the shipped Q-tables and bins: no calibration, no training.
if [ "$REPLICATE" -eq 1 ]; then
  FROM_STAGE="main"
  AGENT_DIR="results/agents"
  BINS="results/agents/state_bins.json"
fi

# Split sizes must be environment variables: the multiprocessing workers rebuild
# the instances themselves and inherit os.environ, not the parent's arguments.
export FYP_TRAIN_INSTANCES="$TRAIN_PROBLEMS"
export FYP_TEST_INSTANCES="$TEST_PROBLEMS"
export PYTHONUNBUFFERED=1

# Same reasoning for the RL stall penalty: GAConfig reads it when a variant is
# rebuilt inside a worker.  Exporting it here covers every stage - calibration,
# training and evaluation - with one value, so an agent can never be trained
# under one reward and evaluated under another.
if [ -n "$RL_PENALTY" ]; then
  export FYP_RL_STALL_PENALTY="$RL_PENALTY"
fi

workers_flag=""
[ "$WORKERS" -gt 0 ] && workers_flag="--workers $WORKERS"
ablation_flag=""
[ "$ABLATION_AGENTS" -eq 1 ] && ablation_flag="--train-ablation-agents"

stage_index() {
  case "$1" in
    instances) echo 0 ;; calibrate) echo 1 ;; train) echo 2 ;; main) echo 3 ;;
    timing) echo 4 ;; analyze) echo 5 ;;
    *) echo "unknown stage: $1" >&2; exit 2 ;;
  esac
}
START_AT=$(stage_index "$FROM_STAGE")

run() {  # run <stage> <description> <command...>
  local stage="$1"; shift
  local what="$1"; shift
  if [ "$(stage_index "$stage")" -lt "$START_AT" ]; then
    echo "[$(date +%H:%M:%S)] -- skipping $stage ($what)"
    return 0
  fi
  echo ""
  echo "=============================================================================="
  echo "[$(date +%H:%M:%S)] STAGE $stage: $what"
  echo "=============================================================================="
  echo "\$ $*"
  if [ "$DRY_RUN" -eq 1 ]; then return 0; fi
  local t0=$SECONDS
  "$@"
  echo "[$(date +%H:%M:%S)] -- $stage finished in $(( (SECONDS - t0) / 60 ))m $(( (SECONDS - t0) % 60 ))s"
}

main() {
  cat <<EOF
==============================================================================
 RL-GA study run
==============================================================================
 mode             : $([ "$REPLICATE" -eq 1 ] && echo "replicate (shipped agents)" || echo "full rebuild")
 control interval : $INTERVAL generations
 problems         : $TRAIN_PROBLEMS train / $TEST_PROBLEMS test
 runs per problem : $RUNS
 generations      : $GENERATIONS      population: $POPULATION
 agent seeds      : $AGENT_SEEDS
 RL stall penalty : ${RL_PENALTY:--0.1}
 ablation agents  : $([ "$ABLATION_AGENTS" -eq 1 ] && echo yes || echo no)
 python           : $PY  ($($PY -c 'import sys; print(sys.version.split()[0])'))
 agents  ->  $AGENT_DIR
 results ->  $RESULTS_DIR
 log     ->  $LOG
 starting at stage: $FROM_STAGE
==============================================================================
EOF

  run instances "instance catalogue and per-instance features" \
    $PY -c "from core.problems import dump_instance_catalogue; \
dump_instance_catalogue('${RESULTS_DIR}/instances.json')"

  run calibrate "RL state bins from the training split" \
    $PY -m train.calibrate_state_bins \
      --generations "$GENERATIONS" --population-size "$POPULATION" \
      --output "$BINS" $workers_flag

  run train "train agents ($RUNS runs x $TRAIN_PROBLEMS problems, interval $INTERVAL)" \
    $PY -m run_all --only train \
      --runs-per-training-instance "$RUNS" \
      --control-interval "$INTERVAL" \
      --agent-seeds $AGENT_SEEDS \
      --generations "$GENERATIONS" --population-size "$POPULATION" \
      --agent-dir "$AGENT_DIR" --bins "$BINS" \
      --results-dir "$RESULTS_DIR" $ablation_flag $workers_flag

  run main "main grid: 11 variants x $TEST_PROBLEMS problems x $RUNS seeds" \
    $PY -m run_all --only main \
      --runs "$RUNS" --control-interval "$INTERVAL" \
      --agent-seeds $AGENT_SEEDS \
      --generations "$GENERATIONS" --population-size "$POPULATION" \
      --agent-dir "$AGENT_DIR" --bins "$BINS" \
      --results-dir "$RESULTS_DIR" $workers_flag

  if [ "$REPLICATE" -eq 0 ]; then
    run timing "sequential wall-clock replication ($TIMING_RUNS seeds)" \
      $PY -m run_all --only timing \
        --timing-runs "$TIMING_RUNS" --control-interval "$INTERVAL" \
        --agent-seeds $AGENT_SEEDS \
        --generations "$GENERATIONS" --population-size "$POPULATION" \
        --agent-dir "$AGENT_DIR" --bins "$BINS" \
        --results-dir "$RESULTS_DIR"
  fi

  run analyze "statistics, report and figures" \
    $PY -m run_all --only analyze figures \
      --results-dir "$RESULTS_DIR"

  echo ""
  echo "=============================================================================="
  echo "[$(date +%H:%M:%S)] RUN COMPLETE in $((SECONDS / 60))m $((SECONDS % 60))s"
  echo "  report : $RESULTS_DIR/analysis/report.md"
  echo "  figures: $RESULTS_DIR/figures/"
  echo "  agents : $AGENT_DIR/"
  echo "  log    : $LOG"
  echo "=============================================================================="
}

main 2>&1 | tee "$LOG"
