#!/bin/bash -l
mamba env create -f environment.yml

conda activate mace-mlmm
# install openmmtools
pip install .

mkdir mace_build

cd mace_build

pip install git+https://github.com/jharrymoore/openmm-ml.git@mace
pip install git+https://github.com/choderalab/mpiplus.git
pip install git+https://github.com/ACEsuit/mace.git@develop

