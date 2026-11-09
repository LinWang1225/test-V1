#!/bin/sh
set -eu
native_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
g++ -std=c++17 -O2 -shared -fPIC "$native_dir/direct_counter.cpp" \
  -o "$native_dir/libdirect_counter.so" \
  -I/usr/local/cuda-12.6/include \
  -I/usr/local/cuda-12.6/extras/CUPTI/include \
  -I/usr/local/cuda-12.6/extras/CUPTI/samples/common \
  -L/usr/local/cuda-12.6/extras/CUPTI/lib64 \
  -Wl,-rpath,/usr/local/cuda-12.6/extras/CUPTI/lib64 -lcupti -lcuda
