# shellcheck shell=bash
#
# Shared Python detection for run_pipeline.sh.
#
# "python" is not reliably on PATH in every Git Bash session on Windows, and a
# name that resolves is not necessarily usable (the Microsoft Store stub in
# WindowsApps resolves but cannot import anything).  Each candidate is therefore
# executed and required to import numpy, which every stage needs.
#
# matplotlib is checked separately and only warned about: it is needed by the
# figures stage alone, so a missing plotting library should not block training
# and analysis that would otherwise run fine.
#
# Sets: PY (the interpreter, possibly with arguments, e.g. "py -3")
# Uses: PYTHON_BIN (explicit override from --python or $PYTHON)

_py_candidates() {
  local candidates=()
  [ -n "${PYTHON_BIN:-}" ] && candidates+=("$PYTHON_BIN")
  candidates+=("python" "python3" "py -3" "py")
  local extra
  for extra in "${LOCALAPPDATA:-}/Python"/pythoncore-*/python.exe \
               "${LOCALAPPDATA:-}/Programs/Python"/Python3*/python.exe \
               "${PROGRAMFILES:-}/Python3"*/python.exe \
               /c/Python3*/python.exe \
               "$HOME"/AppData/Local/Python/pythoncore-*/python.exe \
               "$HOME"/AppData/Local/Programs/Python/Python3*/python.exe; do
    [ -x "$extra" ] && candidates+=("$extra")
  done
  printf '%s\n' "${candidates[@]}"
}

# Why did a candidate fail?  Returns one of: ok | missing | no-numpy
_py_status() {
  # shellcheck disable=SC2086  # intentional: allows candidates like "py -3"
  if ! $1 -c "pass" >/dev/null 2>&1; then
    echo "missing"
  elif ! $1 -c "import numpy" >/dev/null 2>&1; then
    echo "no-numpy"
  else
    echo "ok"
  fi
}

detect_python() {
  local candidate status
  while IFS= read -r candidate; do
    [ -n "$candidate" ] || continue
    status="$(_py_status "$candidate")"
    if [ "$status" = "ok" ]; then
      echo "$candidate"
      return 0
    fi
  done < <(_py_candidates)
  return 1
}

report_python_candidates() {
  local candidate status
  echo "Candidates tried, in order:" >&2
  while IFS= read -r candidate; do
    [ -n "$candidate" ] || continue
    status="$(_py_status "$candidate")"
    case "$status" in
      ok)       printf '  %-64s OK\n'                       "$candidate" >&2 ;;
      missing)  printf '  %-64s not found / will not run\n'  "$candidate" >&2 ;;
      no-numpy) printf '  %-64s runs, but "import numpy" fails\n' "$candidate" >&2 ;;
    esac
  done < <(_py_candidates)
}

require_python() {
  if PY="$(detect_python)"; then
    # matplotlib is only needed for the figures stage - warn, do not block.
    # shellcheck disable=SC2086
    if ! $PY -c "import matplotlib" >/dev/null 2>&1; then
      echo "WARNING: $PY has numpy but not matplotlib." >&2
      echo "         Everything except the figures stage will run." >&2
      echo "         Install it with:  $PY -m pip install matplotlib" >&2
      echo "" >&2
    fi
    export PY
    return 0
  fi

  cat >&2 <<'EOF'
ERROR: no usable Python found.

This project needs an interpreter that can "import numpy".

EOF
  report_python_candidates
  cat >&2 <<EOF

Fix it in one of these ways:

  1. Point at your interpreter explicitly:
       $0 --python "py -3"
       $0 --python /c/Users/you/AppData/Local/Programs/Python/Python312/python.exe

  2. Or export it once for the session:
       export PYTHON="py -3"

  3. If an interpreter runs but has no numpy, install the dependencies:
       py -3 -m pip install numpy matplotlib

To find your interpreters:
       py -0p                      # lists installed versions and paths
       where python                # in cmd / PowerShell
       ls "\$LOCALAPPDATA/Programs/Python"
EOF
  return 1
}
