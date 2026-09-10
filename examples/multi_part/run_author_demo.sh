#!/usr/bin/env bash
# Launch the interactive articulated-model authoring demo.
#
#   run_author_demo.sh live                 # fixed RealSense, move the object
#   run_author_demo.sh replay <seq-dir>     # causal replay of a recording
#   run_author_demo.sh prep   <render> <out>  # render dir -> tracker input
#
# The env prefix is mandatory: gsplat JIT-compiles its CUDA extension on import.
# CUDA_HOME + ninja on PATH are not enough here -- the build also needs the CUDA
# headers on CPATH, and the arch flag set for this GPU. Per the 2026-09-10 build
# note in doc/agent_dialogue.md:
#   CPATH               = $P/targets/x86_64-linux/include:$P/include (+ existing)
#   TORCH_CUDA_ARCH_LIST= 8.9 for the RTX 4070 Ti (explicit override honoured)
#   MAX_JOBS            = 2 (conservative build parallelism)
# Changing the arch flag may rebuild the extension once; matching runs reuse the
# cache. Do not launch this while another process is mid-build.
set -euo pipefail

P="${POINT2POSE_ENV:-$HOME/miniconda3/envs/point2pose_model}"
PORT="${PORT:-8088}"
CONFIG="${CONFIG:-reprojection_split.yaml}"
OUT="${OUT:-results/author_demo_v1/$(date +%Y%m%d)}"
CKPT="${CKPT:-checkpoints/tapir/causal_bootstapir_checkpoint.pt}"
CPATH_BUILD="$P/targets/x86_64-linux/include:$P/include${CPATH:+:$CPATH}"
ARCH="${TORCH_CUDA_ARCH_LIST:-8.9}"
RUN=(env CUDA_HOME="$P" PATH="$P/bin:$PATH" CPATH="$CPATH_BUILD" \
     TORCH_CUDA_ARCH_LIST="$ARCH" MAX_JOBS="${MAX_JOBS:-2}" "$P/bin/python" -u)

mode="${1:-live}"; shift || true
case "$mode" in
  live)
    # Left-click (or --bbox-prompt drag) the object in the OpenCV window; SAM2
    # segments it, the tracker builds the model, the browser shows it live.
    exec "${RUN[@]}" -m examples.multi_part.author_demo \
      --config "$CONFIG" --bbox-prompt --port "$PORT" --output "$OUT" "$@" ;;
  replay)
    seq="${1:?usage: run_author_demo.sh replay <seq-dir>}"; shift
    exec "${RUN[@]}" -m examples.multi_part.author_demo \
      --seq-dir "$seq" --config "$CONFIG" --checkpoint "$CKPT" \
      --port "$PORT" --output "$OUT" "$@" ;;
  prep)
    render="${1:?usage: run_author_demo.sh prep <render-dir> <out-dir>}"
    out="${2:?usage: run_author_demo.sh prep <render-dir> <out-dir>}"
    exec "${RUN[@]}" -m examples.multi_part.prep_demo_input "$render" "$out" ;;
  *)
    echo "unknown mode: $mode (use live|replay|prep)" >&2; exit 2 ;;
esac
