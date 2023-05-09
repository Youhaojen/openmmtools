# unit tests for mace-md entrypoints
# uses pytest to run    
import os
import pytest

from openmmtools.openmm_torch.hybrid_md import PureSystem, MixedSystem
import torch
torch.set_default_dtype(torch.float64)

TEST_DIR = "examples/example_data"
print(TEST_DIR)
JUNK_DIR = os.path.join(TEST_DIR, "junk")
model_path = os.path.join(TEST_DIR,"MACE_SPICE_larger.model")



def test_nonperiodic_pure_mace_torch_nl():

    system=PureSystem(
      	ml_mol=os.path.join(TEST_DIR, "ejm_31.sdf"),
        model_path=model_path,
        potential="mace",
        output_dir=JUNK_DIR,
        temperature=298,
        nl="torch"
    )

    system.run_mixed_md(
        steps=100, interval=25, output_file="output_sm_torch.pdb", restart=False,
    )


    # check the output file exists and is larger than 0 bytes
    assert os.path.exists(os.path.join(JUNK_DIR,"output_sm_torch.pdb"))
    assert os.path.getsize(os.path.join(JUNK_DIR,"output_sm_torch.pdb")) > 0


    
def test_nonperiodic_pure_mace_nnpops_nl():

    system=PureSystem(
      	ml_mol=os.path.join(TEST_DIR, "ejm_31.sdf"),
        model_path=model_path,
        potential="mace",
        output_dir=JUNK_DIR,
        temperature=298,
        nl="nnpops",
        minimise=False
    )

    system.run_mixed_md(
        steps=100, interval=25, output_file="output_sm_nnpops.pdb", restart=False,
    )


    # check the output file exists and is larger than 0 bytes
    assert os.path.exists(os.path.join(JUNK_DIR,"output_sm_nnpops.pdb"))
    assert os.path.getsize(os.path.join(JUNK_DIR,"output_sm_nnpops.pdb")) > 0


def test_periodic_pure_mace_torch_nl():

    system=PureSystem(
      	ml_mol=os.path.join(TEST_DIR, "waterbox.xyz"),
        model_path=model_path,
        potential="mace",
        output_dir=JUNK_DIR,
        temperature=298,
        nl="torch"
    )

    system.run_mixed_md(
        steps=100, interval=25, output_file="output_water_torch.pdb", restart=False,
    )
    assert os.path.exists(os.path.join(JUNK_DIR,"output_water_torch.pdb"))
    assert os.path.getsize(os.path.join(JUNK_DIR,"output_water_torch.pdb")) > 0


def test_periodic_pure_mace_nnpops_nl():

    system=PureSystem(
      	ml_mol=os.path.join(TEST_DIR, "waterbox.xyz"),
        model_path=model_path,
        potential="mace",
        output_dir=JUNK_DIR,
        temperature=298,
        nl="nnpops"
    )

    system.run_mixed_md(
        steps=100, interval=25, output_file="output_water_nnpops.pdb", restart=False,
    )
    assert os.path.exists(os.path.join(JUNK_DIR,"output_water_nnpops.pdb"))
    assert os.path.getsize(os.path.join(JUNK_DIR,"output_water_nnpops.pdb")) > 0


def test_periodic_pure_mace_nnpops_nl_npt():

    system=PureSystem(
      	ml_mol=os.path.join(TEST_DIR, "waterbox.xyz"),
        model_path=model_path,
        potential="mace",
        output_dir=JUNK_DIR,
        temperature=298,
        nl="nnpops",
        pressure=1.0
    )

    system.run_mixed_md(
        steps=100, interval=25, output_file="output_water_nnpops_npt.pdb", restart=False,
    )
    assert os.path.exists(os.path.join(JUNK_DIR,"output_water_nnpops_npt.pdb"))
    assert os.path.getsize(os.path.join(JUNK_DIR,"output_water_nnpops_npt.pdb")) > 0
    
def test_hybrid_system_nnpops_nl():
    system = MixedSystem(
        file=os.path.join(TEST_DIR, "ejm_31.sdf"),
        ml_mol=os.path.join(TEST_DIR, "ejm_31.sdf"),
        model_path=model_path,
        potential="mace",
        output_dir=JUNK_DIR,
        temperature=298,
        nl="nnpops",
        nnpify_type="resname",
        resname="UNK",
        minimise=False
    )

    system.run_mixed_md(
        steps=100, interval=25, output_file="output_hybrid_nnpops.pdb", restart=False,)
    assert os.path.exists(os.path.join(JUNK_DIR,"output_hybrid_nnpops.pdb"))
    assert os.path.getsize(os.path.join(JUNK_DIR,"output_hybrid_nnpops.pdb")) > 0


def test_hybrid_system_torch_nl():

    system = MixedSystem(
        file=os.path.join(TEST_DIR, "ejm_31.sdf"),
        ml_mol=os.path.join(TEST_DIR, "ejm_31.sdf"),
        model_path=model_path,
        potential="mace",
        output_dir=JUNK_DIR,
        temperature=298,
        nl="torch",
        nnpify_type="resname",
        resname="UNK",
        minimise=False
    )

    system.run_mixed_md(
        steps=100, interval=25, output_file="output_hybrid_torch.pdb", restart=False,)
    assert os.path.exists(os.path.join(JUNK_DIR,"output_hybrid_torch.pdb"))
    assert os.path.getsize(os.path.join(JUNK_DIR,"output_hybrid_torch.pdb")) > 0


