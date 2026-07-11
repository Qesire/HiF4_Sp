#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "[ERROR] line=$LINENO cmd=$BASH_COMMAND exit=$?" >&2' ERR

ROOT="${ROOT:-/home/scc/pb23061276/projects/wan-quant}"
HIF4_REPO="${HIF4_REPO:-$ROOT/code/HiF4_Sp}"
SEED="${SEED:-42}"
START_OFFSET="${START_OFFSET:-10}"
N_PER_DIM="${N_PER_DIM:-20}"
RUN_ROOT="${RUN_ROOT:-$ROOT/experiments/hif4_rtn_w4a4_scale60_seed${SEED}_array}"
MERGED_GEN_DIR="$RUN_ROOT/generation/hif4_rtn_w4a4_seed${SEED}"
MERGED_FULL_INFO="$RUN_ROOT/hif4_scale60_start${START_OFFSET}_n${N_PER_DIM}_full_info.json"
DIMS=(i2v_subject i2v_background camera_motion)

export PYTHONPATH="$HIF4_REPO/tools/vbench_i2v_hif4:${PYTHONPATH:-}"
ARGS=()
for dim in "${DIMS[@]}"; do
  DONE="$RUN_ROOT/status/${dim}.done.txt"
  GEN="$RUN_ROOT/shards/$dim/generated"
  INFO="$RUN_ROOT/shards/$dim/full_info/${dim}_start${START_OFFSET}_n${N_PER_DIM}_full_info.json"
  test -s "$DONE"
  test -s "$INFO"
  COUNT=$(find "$GEN" -maxdepth 1 -type f -name '*.mp4' | wc -l)
  echo "$dim MP4_COUNT=$COUNT"
  test "$COUNT" -eq "$N_PER_DIM"
  ARGS+=(--shard "$dim=$GEN" --full-info "$INFO")
done

rm -rf "$MERGED_GEN_DIR"
mkdir -p "$MERGED_GEN_DIR"
python -m hif4_vbench_i2v.merge_scale60_outputs \
  "${ARGS[@]}" \
  --out-dir "$MERGED_GEN_DIR" \
  --merged-full-info "$MERGED_FULL_INFO" \
  --copy-mode physical

MERGED_MP4_COUNT=$(find "$MERGED_GEN_DIR" -maxdepth 1 -type f -name '*.mp4' | wc -l)
MERGED_SYMLINKS=$(find "$MERGED_GEN_DIR" -type l | wc -l)
echo "MERGED_MP4_COUNT=$MERGED_MP4_COUNT"
echo "MERGED_SYMLINKS=$MERGED_SYMLINKS"
test "$MERGED_MP4_COUNT" -eq $((3 * N_PER_DIM))
test "$MERGED_SYMLINKS" -eq 0

cat > "$RUN_ROOT/status/all_shards_merged.done.txt" <<EOF
DONE_AT=$(date -Is)
SEED=$SEED
MERGED_MP4_COUNT=$MERGED_MP4_COUNT
MERGED_GEN_DIR=$MERGED_GEN_DIR
MERGED_FULL_INFO=$MERGED_FULL_INFO
EOF

echo "HIF4_RTN_W4A4_SCALE60_ARRAY_MERGE_DONE"
