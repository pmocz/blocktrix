#!/bin/bash

# This script sets up a Python virtual environment for the blocktrix project on the Rusty cluster.

# Call it from the root of the blocktrix repository:

#   VENVDIR=/path/to/venv_dir scripts/setup_rusty_venv.sh [--update]

# Check that the VENVDIR environment variable is set. If not, print an error message and exit.
if [ -z "$VENVDIR" ]; then
  echo "Error: VENVDIR environment variable is not set."
  exit 1
fi

# Create a virtual environment in the specified directory.
# If the first argument is '--update', do not remove any existing virtual environment.
if [ "$1" != "--update" ]; then
  rm -fr $VENVDIR/blocktrix-venv
fi

# Load the necessary Python module and create a virtual environment with system site packages.
module purge
module load python/3.12.9
python -m venv --system-site-packages $VENVDIR/blocktrix-venv

# Activate the virtual environment and upgrade pip.
source $VENVDIR/blocktrix-venv/bin/activate
pip install --upgrade pip

# Install the blocktrix package with CUDA 12 support.
pip install .[cuda12]
