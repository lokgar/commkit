#!/usr/bin/env bash
# Run the GPU tests of commkit's kernels under NVIDIA compute-sanitizer.
#
#   memcheck   out-of-bounds and misaligned accesses
#   racecheck  shared-memory races
#   synccheck  barrier misuse
#   initcheck  reads of uninitialized device memory
#
# Usage: scripts/sanitize.sh [tool ...]     (default: all four)
#
# The sanitizer is not a dependency. Install its wheel, matching the CUDA major
# version of cupy-cuda13x, into a scratch directory (or set SANITIZER):
#
#   uv pip install --target /tmp/sanitizer "nvidia-cuda-sanitizer-api==13.*"
#
# The 13.2 and 13.3 wheels lack the launcher helpers (TreeLauncherSubreaper,
# TreeLauncherTargetLdPreloadHelper) and stop every target, even /bin/echo, with
# "Target application terminated before first instrumented API call". 13.4 has
# them and runs on a 13.3 driver; for an older driver take NVIDIA's complete
# redistributable archive:
# developer.download.nvidia.com/compute/cuda/redist/cuda_sanitizer_api/
#
# WSL2 needs the Windows GPU debugger interface; without it every run stops at
# "Error: Device not supported" (older: "Failed to initialize WDDM debugger
# interface"). Enable it once from an elevated Windows prompt, then `wsl --shutdown`:
#
#   reg add "HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger" /v EnableInterface /t REG_DWORD /d 1 /f
#
# Windows kills a kernel running longer than TdrDelay seconds
# (HKLM\System\CurrentControlSet\Control\GraphicsDrivers; 8 on the reference
# machine, which the pass stays under); raise it if a sanitized launch times out.
#
# The tests run 10-100x slower; the whole pass takes about 50 minutes on the
# reference machine.
#
# Do not add --leak-check full: CuPy's pool holds device memory until exit, so
# it finds nothing in commkit, and every CuPy process then exits with SIGSEGV
# after the summary.
set -euo pipefail

S=${SANITIZER:-/tmp/sanitizer/nvidia/cu13/bin/compute-sanitizer}
PY=.venv/bin/python
TOOLS=("$@")
[[ ${#TOOLS[@]} -eq 0 ]] && TOOLS=(memcheck racecheck synccheck initcheck)

# racecheck and synccheck concern shared memory and barriers, which only the
# .cu kernels and the commkit_* ElementwiseKernels use, so they are filtered to
# those. Unfiltered, racecheck tracks every CuPy and cuFFT launch of the
# 100k-symbol graph tests and grows by about 0.1 GB of host memory per second.
# memcheck and initcheck run unfiltered: a CuPy kernel reading a buffer commkit
# never filled is a commkit bug.
OWN=(--kernel-name kns=commkit_ --kernel-name kns=bps_ --kernel-name kns=cs_block
     --kernel-name kns=llr --kernel-name kns=iir_ --kernel-name kns=selftest
     --kernel-name kns=ldpc --kernel-name kns=algebraic --kernel-name kns=interpolate)

FILES=(
  tests/test_cuda_infra.py
  tests/equalization/test_cs_kernel.py tests/equalization/test_bps_kernel.py tests/equalization/test_blind.py
  tests/mapping/test_llr.py tests/mapping/test_gray.py tests/recovery/test_bps.py
  tests/test_filtering.py tests/test_fiber.py tests/impairments/test_fiber_channel_nonlinear.py
  tests/coding/test_ldpc_kernel.py tests/coding/test_algebraic.py
  tests/test_interpolation.py tests/impairments/test_frontend.py tests/timing/test_clock_recovery.py
)
# One process per file, and one per test for test_block.py: under the sanitizer
# each CUDA-graph test of the block equalizers gets slower the more tests ran
# before it in the process (one took 10 minutes after 26 others, 61 s alone).
mapfile -t BLOCK < <("$PY" -m pytest --device=gpu -q --collect-only tests/equalization/test_block.py | grep '::')

for tool in "${TOOLS[@]}"; do
  filter=()
  [[ $tool == racecheck || $tool == synccheck ]] && filter=("${OWN[@]}")
  for target in "${FILES[@]}" "${BLOCK[@]}"; do
    echo "== $tool $target"
    "$S" --tool "$tool" "${filter[@]}" "$PY" -m pytest --device=gpu -p no:cacheprovider -q "$target" \
      2>&1 | grep -E "SUMMARY|passed|failed" || true
  done
done
