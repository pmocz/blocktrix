#!/bin/bash

# check that VENVDIR is set
if [ -z "$VENVDIR" ]; then
  echo "Error: VENVDIR environment variable is not set."
  exit 1
fi

# Create virtual environment on rusty
rm -fr $VENVDIR/blocktrix-venv

module purge
module load python/3.12.9
python -m venv --system-site-packages $VENVDIR/blocktrix-venv
source $VENVDIR/blocktrix-venv/bin/activate
pip install --upgrade pip
pip install .[cuda12]
