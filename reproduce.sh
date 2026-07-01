#!/usr/bin/env bash
# reproduce.sh — single command per spec Section 10.3
set -e
docker build -t redrob-ranker .
docker run --rm -v "$(pwd)/data:/app/data" redrob-ranker
