#!/bin/sh
set -eu

bundle_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

mkdir -p /root/dataset
ln -sfn "$bundle_dir/src/vllm-hyrex" /root/exp-vllm-single-forward
ln -sfn "$bundle_dir/src/lmcache-hyrex" /root/exp-lmcache-single-forward
ln -sfn "$bundle_dir/src/vllm-native" /root/exp-vllm-pristine-3way
ln -sfn "$bundle_dir/src/lmcache-native" /root/exp-lmcache-pristine-3way
ln -sfn "$bundle_dir/results" /root/hyrex_results
ln -sfn "$bundle_dir/paper" /root/hyrex-www-paper
ln -sfn "$bundle_dir/datasets/hyrex_traces" /root/dataset/hyrex_traces
ln -sfn "$bundle_dir/datasets/BFCL" /root/dataset/BFCL

printf '%s\n' "HyRex source/result compatibility links created."
printf '%s\n' "Install the runtime and model weights separately before running experiments."
