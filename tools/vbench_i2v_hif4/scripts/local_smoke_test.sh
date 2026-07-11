#!/usr/bin/env bash
set -euo pipefail

# 无需真实 VBench/GPU。验证严格 exact 模式与显式 replicate-base 兼容模式。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"
TMP="${TMPDIR:-/tmp}/hif4_vbench_i2v_smoke_$$"
rm -rf "$TMP"
mkdir -p "$TMP/template/i2v_subject_background/videos_quant_sb" "$TMP/template/i2v_camera_only/videos_quant_camera"
mkdir -p "$TMP/generated_broken" "$TMP/generated_full" "$TMP/results/hif4_seed42/quant/i2v_subject"

for base in "cat on grass" "dog near lake"; do
  echo "one base video for $base" > "$TMP/generated_broken/${base}-0.mp4"
  for i in 0 1 2 3 4; do
    echo "generated full $base repeat $i" > "$TMP/generated_full/${base}-${i}.mp4"
    echo "template $base repeat $i" > "$TMP/template/i2v_subject_background/videos_quant_sb/${base}-${i}.mp4"
  done
done
for base in "camera pans left"; do
  echo "one base video for $base" > "$TMP/generated_broken/${base}-0.mp4"
  for i in 0 1 2 3 4; do
    echo "generated full $base repeat $i" > "$TMP/generated_full/${base}-${i}.mp4"
    echo "template $base repeat $i" > "$TMP/template/i2v_camera_only/videos_quant_camera/${base}-${i}.mp4"
  done
done

echo '[]' > "$TMP/template/i2v_subject_background/i2v_subject_full_info.json"
echo '[]' > "$TMP/template/i2v_camera_only/camera_motion_full_info.json"

python -m hif4_vbench_i2v.preflight --skip-import --scratch-dir "$TMP/scratch" --json-report "$TMP/preflight.json"

# 默认 exact 模式必须拒绝只有 repeat-0 的目录。
if python -m hif4_vbench_i2v.build_eval_inputs \
  --template-case "$TMP/template" \
  --generated-dir "$TMP/generated_broken" \
  --out-case "$TMP/evaluation_inputs/unexpected" \
  --copy-mode physical >/tmp/hif4_vbench_i2v_unexpected_success.log 2>&1; then
  cat /tmp/hif4_vbench_i2v_unexpected_success.log
  echo "ERROR: exact mode unexpectedly accepted repeat-0 only" >&2
  exit 1
fi

# 严格五次独立输入。
python -m hif4_vbench_i2v.build_eval_inputs \
  --template-case "$TMP/template" \
  --generated-dir "$TMP/generated_full" \
  --out-case "$TMP/evaluation_inputs/exact" \
  --repeat-policy exact \
  --copy-mode physical
python -m hif4_vbench_i2v.validate_case_input \
  --case-input "$TMP/evaluation_inputs/exact" \
  --expected-sb 10 \
  --expected-camera 5 \
  --forbid-symlink

# 历史 scale60 兼容输入，必须显式确认且验收时允许相同 hash。
python -m hif4_vbench_i2v.build_eval_inputs \
  --template-case "$TMP/template" \
  --generated-dir "$TMP/generated_broken" \
  --out-case "$TMP/evaluation_inputs/replicated" \
  --repeat-policy replicate-base \
  --acknowledge-replicated-repeats \
  --copy-mode physical
python -m hif4_vbench_i2v.validate_case_input \
  --case-input "$TMP/evaluation_inputs/replicated" \
  --expected-sb 10 \
  --expected-camera 5 \
  --allow-identical-repeat-files \
  --forbid-symlink

mkdir -p "$TMP/results/hif4_seed42/quant/i2v_subject"
echo '{"i2v_subject": 0.9}' > "$TMP/results/hif4_seed42/quant/i2v_subject/hif4_seed42_quant_i2v_subject_eval_results.json"
python -m hif4_vbench_i2v.scan_missing --out-base "$TMP/results" --cases hif4_seed42 --modes quant --dims i2v_subject

python -m py_compile hif4_vbench_i2v/wan_adapter/*.py

echo "LOCAL_SMOKE_TEST_OK tmp=$TMP"
