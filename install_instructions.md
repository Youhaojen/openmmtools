## Manual Installation steps

If the installation script fails on your system, here are some manual steps you can take to replicate the environment.

```
conda install mamba -c conda-forge
```
- If you are installing on a headnode, override the virtual cuda package to match the compute node CUDA version.
```
export CONDA_OVERRIDE_CUDA=11.8
mamba env create -f mace-openmm.yml

```
- Install mace and a MPI util package from GH
```
pip install git+https://github.com/choderalab/mpiplus.git
pip install git+https://github.com/ACEsuit/mace.git
pip install git+https://github.com/jharrymoore/openmm-ml.git@mace
pip install git+https://github.com/jharrymoore/openmmtools.git@development
```
- Tested with the following command
```
wget https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_small.model
mace-md -f ejm_31.sdf --ml_mol ejm_31.sdf --model_path MACE-OFF23_small.model --output_dir md_test --nl torch_nl --steps 1000 --minimiser ase --dtype float64 --remove_cmm --unwrap
```




