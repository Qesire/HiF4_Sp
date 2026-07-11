#!/usr/bin/env bash
#SBATCH -J hif4-rtn-s60
#SBATCH -p Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --gres=gpu:A100:1
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH -t 08:00:00
#SBATCH --array=0-2%2
#SBATCH -o logs/hif4-rtn-s60-%A_%a.out
#SBATCH -e logs/hif4-rtn-s60-%A_%a.err

set -Eeuo pipefail
trap 'echo "[ERROR] line=$LINENO cmd=$BASH_COMMAND exit=$?" >&2' ERR
unset PWD OLDPWD || true
cd "${SAFE_CWD:-/home/scc/pb23061276}"
pwd -P >/dev/null

ROOT="${ROOT:-/home/scc/pb23061276/projects/wan-quant}"
WAN_REPO="${WAN_REPO:-$ROOT/code/Wan2.2-I2V-A14B-W4A4}"
HIF4_REPO="${HIF4_REPO:-$ROOT/code/HiF4_Sp}"
CKPT_DIR="${CKPT_DIR:-$ROOT/weights/Wan2.2-I2V-A14B-bf16}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/data/VBench/vbench2_beta_i2v}"
FULL_INFO_JSON="${FULL_INFO_JSON:-$VBENCH_ROOT/vbench2_i2v_full_info.json}"

START_OFFSET="${START_OFFSET:-10}"
N_PER_DIM="${N_PER_DIM:-20}"
SEED="${SEED:-42}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID missing}"

case "$TASK_ID" in
  0) DIM=i2v_subject ;;
  1) DIM=i2v_background ;;
  2) DIM=camera_motion ;;
  *) echo "unknown task id: $TASK_ID" >&2; exit 2 ;;
esac

RUN_ROOT="${RUN_ROOT:-$ROOT/experiments/hif4_rtn_w4a4_scale60_seed${SEED}_array}"
SHARD_ROOT="$RUN_ROOT/shards/$DIM"
GEN_DIR="$SHARD_ROOT/generated"
FULL_INFO_DIR="$SHARD_ROOT/full_info"
SHARD_FULL_INFO="$FULL_INFO_DIR/${DIM}_start${START_OFFSET}_n${N_PER_DIM}_full_info.json"
SETUP_REPORT="$SHARD_ROOT/${DIM}_hif4_setup_report.json"
RUNTIME_REPORT="$SHARD_ROOT/${DIM}_hif4_runtime_report.json"
MANIFEST="$SHARD_ROOT/${DIM}_manifest.jsonl"
DONE="$RUN_ROOT/status/${DIM}.done.txt"
mkdir -p "$GEN_DIR" "$FULL_INFO_DIR" "$RUN_ROOT/status" "$ROOT/logs"

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${HIF4_ENV:-hif4}"
set -u
export PYTHONNOUSERSITE=1
export PYTHONPATH="$HIF4_REPO/tools/vbench_i2v_hif4:$WAN_REPO:${PYTHONPATH:-}"
export HIF4_MAIN_REPO="$HIF4_REPO"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

python -m hif4_vbench_i2v.build_scale60_full_info \
  --full-info-json "$FULL_INFO_JSON" \
  --out-dir "$FULL_INFO_DIR" \
  --dims "$DIM" \
  --start-offset "$START_OFFSET" \
  --n-per-dim "$N_PER_DIM"

test -s "$SHARD_FULL_INFO"

count_mp4() {
  find "$GEN_DIR" -maxdepth 1 -type f -name '*.mp4' | wc -l
}

if [[ -s "$DONE" ]] && [[ "$(count_mp4)" -eq "$N_PER_DIM" ]]; then
  echo "[SKIP] already complete DIM=$DIM"
  cat "$DONE"
  exit 0
fi
rm -f "$DONE"

cd "$WAN_REPO"
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "ATTEMPT=$attempt DIM=$DIM SEED=$SEED"
  rm -f "$MANIFEST" "$SETUP_REPORT" "$RUNTIME_REPORT"
  set +e
  python tools/generate_vbench_i2v_svdquant.py \
    --mode hif4 \
    --hif4_main_repo "$HIF4_REPO" \
    --hif4_report_json "$SETUP_REPORT" \
    --hif4_runtime_report_json "$RUNTIME_REPORT" \
    --full_info_json "$SHARD_FULL_INFO" \
    --vbench_root "$VBENCH_ROOT" \
    --ckpt_dir "$CKPT_DIR" \
    --output_dir "$GEN_DIR" \
    --manifest "$MANIFEST" \
    --dimensions "$DIM" \
    --samples_per_prompt 1 \
    --seed "$SEED" \
    --overwrite \
    --size "${VIDEO_SIZE:-832*480}" \
    --frame_num "${FRAME_NUM:-61}" \
    --sample_steps "${SAMPLE_STEPS:-4}" \
    --sample_shift "${SAMPLE_SHIFT:-5.0}" \
    --sample_guide_scale "${SAMPLE_GUIDE_SCALE:-5.0}" \
    --device_id 0 \
    --offload_model true \
    --t5_cpu \
    --convert_model_dtype \
    --freeze_condition_latent
  RC=$?
  set -e

  MP4_COUNT=$(count_mp4)
  MANIFEST_LINES=0
  [[ -s "$MANIFEST" ]] && MANIFEST_LINES=$(wc -l < "$MANIFEST")
  echo "ATTEMPT=$attempt RC=$RC MP4_COUNT=$MP4_COUNT MANIFEST_LINES=$MANIFEST_LINES"

  if [[ "$RC" -eq 0 && "$MP4_COUNT" -eq "$N_PER_DIM" ]]; then
    cat > "$DONE" <<EOF
DONE_AT=$(date -Is)
DIM=$DIM
SEED=$SEED
MP4_COUNT=$MP4_COUNT
MANIFEST_LINES=$MANIFEST_LINES
START_OFFSET=$START_OFFSET
N_PER_DIM=$N_PER_DIM
ATTEMPT=$attempt
MODE=hif4_rtn_w4a4
GEN_DIR=$GEN_DIR
SHARD_FULL_INFO=$SHARD_FULL_INFO
EOF
    echo "HIF4_RTN_W4A4_SHARD_DONE DIM=$DIM"
    exit 0
  fi
  [[ "$attempt" -lt "$MAX_ATTEMPTS" ]] && sleep $((attempt * 10))
done

echo "HIF4_RTN_W4A4_SHARD_FAILED DIM=$DIM" >&2
exit 30
