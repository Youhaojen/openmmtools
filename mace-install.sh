mamba create -f environment.yml

# install openmmtools
pip install .

mkdir build

cd build

# git clone https://github.com/jharrymoore/torch_nl.git
# cd torch_nl
# pip install .
# cd ..

pip install git+https://github.com/jharrymoore/openmm-ml.git@mace
git clone https://github.com/ACEsuit/mace.git
cd mace
pip install .

