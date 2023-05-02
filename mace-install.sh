#!/bin/bash -l
mamba env create -f environment.yml

conda activate mace-mlmm
# install openmmtools
pip install .

mkdir build

cd build

pip install git+https://github.com/jharrymoore/openmm-ml.git@mace
pip install git+https://github.com/choderalab/mpiplus.git
git clone https://github.com/ACEsuit/mace.git
cd mace
pip install .

