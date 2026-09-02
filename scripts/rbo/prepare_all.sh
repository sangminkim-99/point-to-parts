#!/usr/bin/env bash
# Prepare every downloaded RBO archive for RBOReader. Idempotent, so it can be
# re-run while fetch_rbo.sh is still downloading.
set -u
REPO="/home/smkim/workspace/code/point-to-pose-model-based"
ROOT="/home/smkim/workspace/dataset/RBO"
OUT="$ROOT/sequences"
mkdir -p "$OUT"
for a in "$ROOT"/raw/*_o.tar.gz; do
  [ -e "$a" ] || continue
  name="$(basename "$a" .tar.gz)"
  [ -d "$OUT/$name/depth_png" ] && continue
  python "$REPO/scripts/rbo/prepare_rbo_sequence.py" "$a" --out-root "$OUT" --quiet
done
echo "DONE $(date -Is)"; du -sh "$OUT"
