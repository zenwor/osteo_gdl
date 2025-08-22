#!/bin/bash

# This script will set up the Conda environment and install the dependencies

export PROJECT_NAME="osteo_gdl"
CONDA_ENV_NAME="${PROJECT_NAME}_venv"
PYTHON_VERSION="3.9"

# Check if the conda environment exists
if conda env list | grep -q "$CONDA_ENV_NAME"; then
    echo "Conda environment '$CONDA_ENV_NAME' already exists. Updating it..."
    conda env update --name "$CONDA_ENV_NAME" --file environment.yaml --prune
else
    echo "Conda environment '$CONDA_ENV_NAME' not found. Creating it from environment.yaml..."
    conda env create --name "$CONDA_ENV_NAME" --file environment.yaml
fi

# Activate the environment
conda activate "$CONDA_ENV_NAME"

echo "Environment setup complete."

# General environment variables
export PROJ_ROOT=$PWD
export SRC_PATH="$PWD/osteo_gdl"
export DATA_PATH="$PROJ_ROOT/data/"
export PYTHONPATH=${PROJ_ROOT}
