#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python worker.py --mode tune &
CUDA_VISIBLE_DEVICES=1 python worker.py --worker &
CUDA_VISIBLE_DEVICES=2 python worker.py --worker &
CUDA_VISIBLE_DEVICES=3 python worker.py --worker &
CUDA_VISIBLE_DEVICES=4 python worker.py --worker &
CUDA_VISIBLE_DEVICES=5 python worker.py --worker &
CUDA_VISIBLE_DEVICES=6 python worker.py --worker &
CUDA_VISIBLE_DEVICES=7 python worker.py --worker &
