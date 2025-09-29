#!/usr/bin/env bash
set -e
# Render will run this
uvicorn app:app --host 0.0.0.0 --port $PORT
