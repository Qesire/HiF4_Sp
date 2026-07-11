#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "[ERROR] line=$LINENO cmd=$BASH_COMMAND exit=$?" >&2' ERR

ROOT="${ROOT:-/home/scc/pb23061276/projects/wan-quant}"
HIF4_REPO="${HIF4_REPO:-$ROOT/code/HiF4_Sp}"
INPUT_BASE="${INPUT_BASE:?set INPUT_BASE to the shared evaluation_inputs directory}"
TEMPLATE_PREFIX="${TEMPLATE_PREFIX:-$INPUT_BASE/empty_seed}"
SEEDS="${SEEDS:-42 43 44}"
VALIDATE_MODE="${VALIDATE_MODE:-both}"

export PYTHONPATH="$HIF4_REPO/tools/vbench_i2v_hif4:${PYTHONPATH:-}"

echo "WARNING: replicate-base reproduces the verified 60 -> 200/100 compatibility workflow."
echo "WARNING: it is not five independent samples per image-prompt pair."

for seed in $SEEDS; do
  RUN_ROOT="$ROOT/experiments/hif4_rtn_w4a4_scale60_seed${seed}_array"
  GENERATED="$RUN_ROOT/generation/hif4_rtn_w4a4_seed${seed}"
  TEMPLATE_CASE="${TEMPLATE_PREFIX}${seed}"
  OUT_CASE="$INPUT_BASE/hif4_rtn_w4a4_seed${seed}"

  test -d "$GENERATED"
  test -d "$TEMPLATE_CASE"
  test "$(find "$GENERATED" -maxdepth 1 -type f -name '*.mp4' | wc -l)" -eq 60

  python -m hif4_vbench_i2v.build_eval_inputs \
    --template-case "$TEMPLATE_CASE" \
    --generated-dir "$GENERATED" \
    --out-case "$OUT_CASE" \
    --repeat-policy replicate-base \
    --acknowledge-replicated-repeats \
    --copy-mode physical

  python -m hif4_vbench_i2v.validate_case_input \
    --case-input "$OUT_CASE" \
    --expected-sb 200 \
    --expected-camera 100 \
    --expected-repeats 5 \
    --mode "$VALIDATE_MODE" \
    --allow-identical-repeat-files \
    --forbid-symlink

done

echo "ALL_HIF4_RTN_W4A4_REPLICATED_CASE_INPUTS_READY"
