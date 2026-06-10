#!/usr/bin/env bash
set -euo pipefail

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests
