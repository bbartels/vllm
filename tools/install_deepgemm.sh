#!/bin/bash
# Script to build and/or install DeepGEMM from source
# Default behaviour (no --stage) = build + install (same as original)

set -e

# Defaults
DEEPGEMM_GIT_REPO="https://github.com/deepseek-ai/DeepGEMM.git"
DEEPGEMM_GIT_REF="ea9c5d9270226c5dd7a577c212e9ea385f6ef048"
STAGE="all"               # default: "all" = build then install
CUDA_VERSION=""
WHEEL_PATH=""
OUT_DIR=""

print_help() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --stage STAGE        One of: build | install | all (default: all)
  --ref REF            Git reference to checkout (default: $DEEPGEMM_GIT_REF)
  --cuda-version VER   CUDA version (auto-detected if not provided for build/all)
  --wheel PATH         Path to an existing wheel (for --stage=install)
  --out-dir DIR        Directory to copy built wheel(s) into (for build/all)
  -h, --help           Show this help message

Examples:
  $0 --stage build --out-dir ./artifacts
  $0 --stage install --wheel ./artifacts/deepgemm-*.whl
  $0                  # default = build + install (original behaviour)
EOF
}

# Parse CLI args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      [[ -n "${2:-}" && ! "$2" =~ ^- ]] || { echo "Error: --stage requires an argument." >&2; exit 1; }
      STAGE="$2"; shift 2;;
    --ref)
      [[ -n "${2:-}" && ! "$2" =~ ^- ]] || { echo "Error: --ref requires an argument." >&2; exit 1; }
      DEEPGEMM_GIT_REF="$2"; shift 2;;
    --cuda-version)
      [[ -n "${2:-}" && ! "$2" =~ ^- ]] || { echo "Error: --cuda-version requires an argument." >&2; exit 1; }
      CUDA_VERSION="$2"; shift 2;;
    --wheel)
      [[ -n "${2:-}" && ! "$2" =~ ^- ]] || { echo "Error: --wheel requires a path to a wheel." >&2; exit 1; }
      WHEEL_PATH="$2"; shift 2;;
    --out-dir)
      [[ -n "${2:-}" && ! "$2" =~ ^- ]] || { echo "Error: --out-dir requires a directory path." >&2; exit 1; }
      OUT_DIR="$2"; shift 2;;
    -h|--help)
      print_help; exit 0;;
    *)
      echo "Unknown option: $1" >&2; echo; print_help; exit 1;;
  esac
done

# Validate stage
case "$STAGE" in
  build|install|all) ;;
  *)
    echo "Error: --stage must be one of: build | install | all" >&2
    exit 1
    ;;
esac

# Helpers
detect_cuda_version() {
  if [ -n "$CUDA_VERSION" ]; then
    return
  fi
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    echo "Auto-detected CUDA version: $CUDA_VERSION"
  else
    echo "Warning: Could not auto-detect CUDA version. Please specify with --cuda-version"
    exit 1
  fi
}

check_cuda_requirement() {
  # Requires CUDA >= 12.8
  local ver="$1"
  local major="${ver%%.*}"
  local minor="${ver#${major}.}"; minor="${minor%%.*}"
  echo "CUDA version: $ver (major: $major, minor: $minor)"
  if [ "$major" -lt 12 ] || { [ "$major" -eq 12 ] && [ "$minor" -lt 8 ]; }; then
    echo "Skipping DeepGEMM build/install (requires CUDA 12.8+ but got ${ver})"
    exit 0
  fi
}

do_build() {
  echo "Installing DeepGEMM build dependencies and building wheel..."
  echo "Repository: $DEEPGEMM_GIT_REPO"
  echo "Reference:  $DEEPGEMM_GIT_REF"

  local install_dir
  install_dir=$(mktemp -d)
  trap 'rm -rf "$install_dir"' EXIT

  git clone --recursive --shallow-submodules "$DEEPGEMM_GIT_REPO" "$install_dir/deepgemm"

  echo "🏗️  Building DeepGEMM"
  pushd "$install_dir/deepgemm" >/dev/null

  git checkout "$DEEPGEMM_GIT_REF"

  rm -rf build dist
  rm -rf *.egg-info
  python3 setup.py bdist_wheel

  # Find produced wheels
  local built_wheels
  built_wheels=(dist/*.whl)
  if [ ! -e "${built_wheels[0]}" ]; then
    echo "Error: No wheel produced in dist/." >&2
    exit 1
  fi

  # Optionally copy to OUT_DIR
  if [ -n "$OUT_DIR" ]; then
    mkdir -p "$OUT_DIR"
    cp -v dist/*.whl "$OUT_DIR/"
    echo "Wheel(s) copied to: $OUT_DIR"
    # Print paths for easy piping
    for w in "${built_wheels[@]}"; do
      echo "ARTIFACT: $OUT_DIR/$(basename "$w")"
    done
  else
    # Print absolute path(s) of built wheel(s)
    for w in "${built_wheels[@]}"; do
      echo "ARTIFACT: $PWD/$w"
    done
  fi

  popd >/dev/null
}

install_wheel() {
  local wheel="$1"
  if ! ls $wheel >/dev/null 2>&1; then
    echo "Error: Wheel not found: $wheel" >&2
    exit 1
  fi

  if command -v uv >/dev/null 2>&1; then
    echo "Installing DeepGEMM wheel using uv..."
    if [ -n "$VLLM_DOCKER_BUILD_CONTEXT" ]; then
      uv pip install --system $wheel
    else
      uv pip install $wheel
    fi
  else
    echo "Installing DeepGEMM wheel using pip..."
    python3 -m pip install $wheel
  fi
}

# --- Stage execution ---

case "$STAGE" in
  build)
    detect_cuda_version
    check_cuda_requirement "$CUDA_VERSION"
    do_build
    echo "✅ Build completed (no installation performed)."
    ;;
  install)
    # For install-only, allow providing a wheel; otherwise look for dist/*.whl in CWD.
    if [ -z "$WHEEL_PATH" ]; then
      # Try local dist/*.whl
      if compgen -G "dist/*.whl" > /dev/null; then
        WHEEL_PATH="dist/*.whl"
        echo "No --wheel provided; using local $WHEEL_PATH"
      else
        echo "Error: --wheel is required for --stage install (or run from a directory with dist/*.whl)." >&2
        exit 1
      fi
    fi
    install_wheel "$WHEEL_PATH"
    echo "✅ Install completed."
    ;;
  all)
    detect_cuda_version
    check_cuda_requirement "$CUDA_VERSION"
    # Build, then install the freshly built wheel(s)
    # Capture the last "ARTIFACT:" line to get a wheel path
    artifact_line=$(do_build | tee /dev/stderr | grep -E "ARTIFACT:" | tail -n 1 || true)
    if [ -z "$artifact_line" ]; then
      echo "Error: Could not determine built wheel path." >&2
      exit 1
    fi
    built_path="${artifact_line#ARTIFACT: }"
    install_wheel "$built_path"
    echo "✅ Build and installation completed successfully."
    ;;
esac
