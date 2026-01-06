#!/usr/bin/env bash
VENV_DIR=".venv"
python3 -m venv "$VENV_DIR"
echo "Virtual environment created at $VENV_DIR"
echo "To activate: source $VENV_DIR/bin/activate"
echo "Then install dependencies: pip install -r requirements.txt"
