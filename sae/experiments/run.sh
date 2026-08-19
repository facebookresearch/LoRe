#!/usr/bin/env bash
# Coordinator for the three SAE interpretability experiments. Modular: pick any
# subset; `all` runs every one and CONTINUES past a failure so one broken script
# does not block the others. Each script is self-contained and re-runnable.
#
#   bash run.sh                 # all three
#   bash run.sh all
#   bash run.sh 1               # just experiment 1 (the key control)
#   bash run.sh 2 3             # experiments 2 and 3
#   bash run.sh exp1 exp2       # names also work
#
# Extra flags after a `--` are forwarded to every selected script, e.g.
#   bash run.sh 1 -- --alpha 0            # data-only variant of the control
#   bash run.sh all -- --device cuda:0
#
# Results land in results/exp{1,2,3}/ ; intermediate directions in artifacts/.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PY:-python}"

declare -A SCRIPT=(
  [1]="exp1_shuffled_control.py"
  [2]="exp2_residual_main.py"
  [3]="exp3_feature_space.py"
  [4]="exp4_data_direction.py"
  [5]="exp5_positive_control.py"
)

# split args into selectors (before --) and forwarded flags (after --)
sel=(); fwd=(); seen_dd=0
for a in "$@"; do
  if [[ "$seen_dd" == 1 ]]; then fwd+=("$a"); continue; fi
  if [[ "$a" == "--" ]]; then seen_dd=1; continue; fi
  sel+=("$a")
done
[[ ${#sel[@]} -eq 0 ]] && sel=(all)

# normalize selectors to the ordered list 1 2 3
order=()
for s in "${sel[@]}"; do
  case "$s" in
    all)        order=(1 2 3 4 5); break ;;
    1|exp1)     order+=(1) ;;
    2|exp2)     order+=(2) ;;
    3|exp3)     order+=(3) ;;
    4|exp4)     order+=(4) ;;
    5|exp5)     order+=(5) ;;
    *) echo "unknown target: $s (use: 1|2|3|4|5|exp1..exp5|all)" >&2; exit 2 ;;
  esac
done

echo "=== running experiments: ${order[*]}  (forward: ${fwd[*]:-none}) ==="
declare -A STATUS
for n in "${order[@]}"; do
  script="${SCRIPT[$n]}"
  echo ""
  echo "======================================================================"
  echo "  exp$n -> $script"
  echo "======================================================================"
  if "$PY" "$script" "${fwd[@]}"; then
    STATUS[$n]="ok"
  else
    STATUS[$n]="FAILED (exit $?)"
    echo "!! exp$n failed; continuing with the rest" >&2
  fi
done

echo ""
echo "=== summary ==="
rc=0
for n in "${order[@]}"; do
  echo "  exp$n: ${STATUS[$n]}"
  [[ "${STATUS[$n]}" == ok ]] || rc=1
done
exit "$rc"
