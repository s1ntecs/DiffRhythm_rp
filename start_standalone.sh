#!/usr/bin/env bash
set -e
echo "Worker Initiated"
echo "Starting RunPod Handler"
export PYTHONUNBUFFERED=1
python3 -u /workspace/rp_handler.py
