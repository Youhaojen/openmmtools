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

@pytest.mark.parametrize("remove_cmm", [True, False])
@pytest.mark.parametrize("file", ["ejm_31.sdf", "waterbox.xyz"])
@pytest.mark.parametrize("nl", ["torch", "nnpops"])
def test_pure_mace_md(file, nl, remove_cmm):
    file_stub = file.split(".")[0]
    cmm = "cmm" if remove_cmm else "nocmm"
    system=PureSystem(
      	ml_mol=os.path.join(TEST_DIR, file),
        model_path=model_path,
        potential="mace",
        output_dir=JUNK_DIR,
        temperature=298,
        nl=nl,
        pressure=1.0 if file_stub == "waterbox" else None,
        remove_cmm=remove_cmm,
    )
    output_file = f"output_pure_{file_stub}_{nl}_{cmm}.pdb"
    system.run_mixed_md(
        steps=20, interval=5, output_file=output_file, restart=False,
    )

    # check the output file exists and is larger than 0 bytes
    assert os.path.exists(os.path.join(JUNK_DIR, output_file))
    assert os.path.getsize(os.path.join(JUNK_DIR, output_file)) > 0


@pytest.mark.parametrize("water_model", ["tip3p", "tip4pew"])
@pytest.mark.parametrize("mm_only", [True, False])
@pytest.mark.parametrize("remove_cmm", [True, False])
@pytest.mark.parametrize("nl", ["torch", "nnpops"])
def test_hybrid_system_md(nl, remove_cmm, mm_only, water_model):
    cmm = "cmm" if remove_cmm else "nocmm"
    mm = "mm_only" if mm_only else "mm_and_ml"
    forcefields = [
            "amber/protein.ff14SB.xml",
            "amber14/DNA.OL15.xml",
            "amber/tip3p_standard.xml"
        ] if water_model == "tip3p" else [
            "amber/protein.ff14SB.xml",
            "amber14/DNA.OL15.xml",
            "amber14/tip4pew.xml"
        ]
    system = MixedSystem(
        file=os.path.join(TEST_DIR, "ejm_31.sdf"),
        ml_mol=os.path.join(TEST_DIR, "ejm_31.sdf"),
        model_path=model_path,
        forcefields=forcefields,
        potential="mace",
        output_dir=JUNK_DIR,
        temperature=298,
        nl=nl,
        nnpify_type="resname",
        resname="UNK",
        minimise=False,
        water_model=water_model,

    )
    output_file = f"output_hybrid_{nl}_{cmm}_{mm}.pdb"

    system.run_mixed_md(
        steps=20, interval=5, output_file=output_file, restart=False)
    assert os.path.exists(os.path.join(JUNK_DIR, output_file))
    assert os.path.getsize(os.path.join(JUNK_DIR, output_file)) > 0

# mark as slow test
@pytest.mark.slow
@pytest.mark.parametrize("water_model", ["tip3p"])
@pytest.mark.parametrize("remove_cmm", [True, False])
@pytest.mark.parametrize("nl", ["torch", "nnpops"])
def test_hybrid_system_repex(nl, remove_cmm, water_model):
    cmm = "cmm" if remove_cmm else "nocmm"
    forcefields = [
            "amber/protein.ff14SB.xml",
            "amber14/DNA.OL15.xml",
            "amber/tip3p_standard.xml"
        ] if water_model == "tip3p" else [
            "amber/protein.ff14SB.xml",
            "amber14/DNA.OL15.xml",
            "amber14/tip4pew.xml"
        ]
    system = MixedSystem(
        file=os.path.join(TEST_DIR, "ejm_31.sdf"),
        ml_mol=os.path.join(TEST_DIR, "ejm_31.sdf"),
        model_path=model_path,
        forcefields=forcefields,
        potential="mace",
        output_dir=JUNK_DIR,
        temperature=298,
        nl=nl,
        nnpify_type="resname",
        interpolate=True,
        resname="UNK",
        minimise=False,
        water_model=water_model,

    )

    system.run_repex(
        replicas=2,
        steps=3,
        steps_per_mc_move=100,
        checkpoint_interval=1,
        decouple=False, 
        restart=False)
    assert os.path.exists(os.path.join(JUNK_DIR, "repex.nc"))
    assert os.path.getsize(os.path.join(JUNK_DIR, "repex.nc")) > 0