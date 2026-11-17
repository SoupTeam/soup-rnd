#!/bin/sh
# Cross-compile ds_probe.exe on Linux with MinGW-w64 and record what was built.
#   sh benchmarks/g3/directstorage/build.sh OUT_DIR
set -eu
here=$(cd "$(dirname "$0")" && pwd)
out=${1:?usage: build.sh OUT_DIR}
mkdir -p "$out"
cxx=${CXX:-x86_64-w64-mingw32-g++-posix}
"$cxx" -std=c++17 -O2 -Wall -Wextra -Werror -static -o "$out/ds_probe.exe" "$here/ds_probe.cpp" \
    -ld3d12 -ldxgi -ldxguid -luuid -lpsapi -lshell32
cp "$here/run_g3.ps1" "$out/"
{
    printf 'compiler: %s\n' "$("$cxx" --version | head -1)"
    printf 'ds_probe.cpp sha256: %s\n' "$(sha256sum "$here/ds_probe.cpp" | cut -d' ' -f1)"
    printf 'dstorage_min.h sha256: %s\n' "$(sha256sum "$here/dstorage_min.h" | cut -d' ' -f1)"
    printf 'ds_probe.exe sha256: %s\n' "$(sha256sum "$out/ds_probe.exe" | cut -d' ' -f1)"
    printf 'run_g3.ps1 sha256: %s\n' "$(sha256sum "$out/run_g3.ps1" | cut -d' ' -f1)"
} > "$out/BUILD.txt"
cat "$out/BUILD.txt"
