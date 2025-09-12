#!/bin/bash
# Usage: ./install_nixl.sh [--force] [--jobs N]

FORCE=false
JOBS=1



while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=true
      shift
      ;;
    --jobs)
      if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
        JOBS=$2
        shift 2
      else
        echo "Error: --jobs requires a numeric argument"
        exit 1
      fi
      ;;
    *)
      echo "Usage: $0 [--force] [--jobs N]"
      exit 1
      ;;
  esac
done

SUDO=false
if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    SUDO=true
fi

ARCH=$(uname -m)

ROOT_DIR="/usr/local"
mkdir -p "$ROOT_DIR"
GDR_HOME="$ROOT_DIR/gdrcopy"
UCX_HOME="$ROOT_DIR/ucx"
NIXL_HOME="$ROOT_DIR/nixl"
CUDA_HOME=/usr/local/cuda

export PATH="$GDR_HOME/bin:$UCX_HOME/bin:$NIXL_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$GDR_HOME/lib:$UCX_HOME/lib:$NIXL_HOME/lib/$ARCH-linux-gnu:$LD_LIBRARY_PATH"

TEMP_DIR="nixl_installer"
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

pip install meson ninja pybind11

if [ ! -e "/dev/gdrdrv" ] || [ "$FORCE" = true ]; then
    echo "Installing gdrcopy\n"
    wget https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v2.5.tar.gz
    tar xzf v2.5.tar.gz; rm v2.5.tar.gz
    cd gdrcopy-2.5
    make -j"$JOBS" prefix=$GDR_HOME CUDA=$CUDA_HOME all install
    
    if $SUDO; then
        echo "Running insmod.sh with sudo"
        sudo ./insmod.sh
    else
        echo "Skipping insmod.sh - sudo not available"
        echo "Please run 'sudo ./gdrcopy-2.5/insmod.sh' manually if needed"
    fi
    
    cd ..
else
    echo "Found /dev/gdrdrv. Skipping gdrcopy installation"
fi

if ! command -v ucx_info &> /dev/null || [ "$FORCE" = true ]; then
    echo "Installing UCX"
    wget https://github.com/openucx/ucx/releases/download/v1.18.0/ucx-1.18.0.tar.gz
    tar xzf ucx-1.18.0.tar.gz; rm ucx-1.18.0.tar.gz
    cd ucx-1.18.0
    
    # Checking Mellanox NICs
    MLX_OPTS=""
    if lspci | grep -i mellanox > /dev/null || command -v ibstat > /dev/null; then
        echo "Mellanox NIC detected, adding Mellanox-specific options"
        MLX_OPTS="--with-rdmacm \
                  --with-mlx5-dv \
                  --with-ib-hw-tm"
    fi
    
    ./configure  --prefix=$UCX_HOME                \
                --enable-shared                    \
                --disable-static                   \
                --disable-doxygen-doc              \
                --enable-optimizations             \
                --enable-cma                       \
                --enable-devel-headers             \
                --with-cuda=$CUDA_HOME             \
                --with-dm                          \
                --with-gdrcopy=$GDR_HOME           \
                --with-verbs                       \
                --enable-mt                        \
                $MLX_OPTS
    make -j"$JOBS"
    make -j"$JOBS" install-strip
    
    if $SUDO; then
        echo "Running ldconfig with sudo"
        sudo ldconfig
    else
        echo "Skipping ldconfig - sudo not available"
        echo "Please run 'sudo ldconfig' manually if needed"
    fi

    cd ..
else
    echo "Found existing UCX. Skipping UCX installation"  
fi

if ! command -v nixl_test &> /dev/null || [ "$FORCE" = true ]; then
    echo "Installing NIXL"
    wget https://github.com/ai-dynamo/nixl/archive/refs/tags/0.2.0.tar.gz
    tar xzf 0.2.0.tar.gz; rm 0.2.0.tar.gz
    cd nixl-0.2.0
    meson setup build --prefix=$NIXL_HOME -Ducx_path=$UCX_HOME
    cd build
    ninja -j"$JOBS"
    ninja -j"$JOBS" install
    /bin/command-that-fails

    cd ../..
else
    echo "Found existing NIXL. Skipping NIXL installation"  
fi
