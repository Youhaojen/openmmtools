from argparse import ArgumentParser
import os
import subprocess
from rdkit.Chem.rdmolfiles import MolToXYZFile, MolToMolBlock, MolFromXYZFile
from ase.io import read, write
from ase.io.extxyz import write_extxyz
from tqdm import tqdm
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
import numpy as np
from multiprocessing import Pool
from typing import Tuple
from ase import Atoms
from shutil import rmtree
from tqdm import tqdm
from tempfile import mkstemp
from openmmtools.openmm_torch.utils import initialize_mm_forcefield
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from copy import deepcopy
from openmm.unit import elementary_charge
from shutil import rmtree


# def extract_atomtypes(values: Tuple[int,  Atoms]):
#     # convert ase atoms to rdkit
#     try:
#         (idx, mol) = values
#         old_dir = os.getcwd()
#         output_dir = os.path.join(os.getcwd(), f"mol_processing_{idx}")
#         os.makedirs(output_dir, exist_ok=True)
#         os.chdir(output_dir)
#         with open( "mol.xyz", 'w') as f:
#             write(f, mol)

#         cmd = f"obabel -ixyz mol.xyz -omol2 -O mol.sdf"
#         subprocess.run(cmd, shell=True, check=True, capture_output=True)
#         molecule = Molecule.from_file("mol.sdf")
#         os.chdir(old_dir)
#         rmtree(output_dir)
#         # _, filename = mkstemp(suffix=".xyz")
#         # print(filename)
#         # # remove extra arrays from the mol
#         # # write_out_mol = deepcopy(mol)
#         # # del(write_out_mol.arrays['forces'])
#         # # print(write_out_mol.arrays.keys())
#         # # print(write_out_mol.info.keys())

#         # with open(filename, 'w') as f:
#         #     write_extxyz(f, mol, plain=True)
#         # rdkit_mol = MolFromXYZFile(filename)
#         # smiles = mol.info['smiles']

#         # molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
        
#         forcefield = ForceField("openff_unconstrained-2.0.0.offxml")
#         partial_charges = forcefield.get_partial_charges(molecule)
#         mol.arrays["am1bcc_charges"] = partial_charges
#         print(partial_charges)
#         ff_applied_params = forcefield.label_molecules(molecule.to_topology())
#         # parse the bonds dictionary, want to end up with the smirks of each atom, from its bonded participant
#         atom_smirks = {}
#         for atom, param in dict(ff_applied_params[0]["Bonds"]).items():
#             atom_0, atom_1 = atom
#             for sep in ["-", "=", ":"]:
#                 try:
#                     smirks_0, smirks_1 = param.smirks.split(f']{sep}[')
#                     smirks_1 = smirks_1.replace('[', "").replace(']',"")
#                     smirks_0 = smirks_0.replace('[', "").replace(']',"")
#                     print(atom_0, atom_1)
#                     print(smirks_0, smirks_1)
#                 except:
#                     pass
#             atom_smirks[atom_0] = "[" + smirks_0 + "]"
#             atom_smirks[atom_1] = "[" + smirks_1 + "]"
#         atom_smirks = dict(sorted(atom_smirks.items()))
#         print(atom_smirks)
#         mol.arrays['atomtypes'] = np.array(list(atom_smirks.values()))
#         return mol
#     except Exception as e:
#         print(e)
#         return None






    


def extract_atomtypes(values: Tuple[int, Atoms]):
    (idx, mol) = values
    old_dir = os.getcwd()
    try:
        output_dir = os.path.join(os.getcwd(), f"mol_processing_{idx}")
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)
        with open( "mol.xyz", 'w') as f:
            write(f, mol)

        # write out mol2 file
        # convert mol to mol2
        cmd = f"obabel -ixyz mol.xyz -omol2 -O mol.mol2"
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        cmd = (
            f"antechamber -fi mol2 -i mol.mol2 -c bcc -fo mol2 -o mol_processed.mol2 -dr n"
        )
        subprocess.run(
            cmd, capture_output=True, universal_newlines=True, check=True, shell=True
        )
        # for line in output.stdout.splitlines():
            # print(line)
        cmd = "cat mol_processed.mol2 | grep UNL | awk '{ print $6 } ' > atomtypes.txt"
        subprocess.run(cmd, capture_output=True, universal_newlines=True, check=True, shell=True)
        cmd = "cat mol_processed.mol2 | grep UNL | awk '{ print $9 } ' > partial_charges.txt"
        subprocess.run(cmd, capture_output=True, universal_newlines=True, check=True, shell=True)
        with open("atomtypes.txt", "r") as f:
            # it captures an extra line at the bottom of the mol2 that we don't need
            atomtypes = [l.strip() for l in f.readlines()[:-1]]
        # open the resulting file, strip the
        with open("partial_charges.txt", "r") as f:
            # it captures an extra line at the bottom of the mol2 that we don't need
            partials = [l.strip() for l in f.readlines()[:-1]]

        mol.set_array("atomtypes", np.array(atomtypes))
        mol.set_array("partial_charges", np.array(partials))
        os.chdir(old_dir)
        rmtree(output_dir)
        print("done")
        return mol
    except Exception as e:
        rmtree(output_dir)
        os.chdir(old_dir)
        print(e)


def main():
    parser = ArgumentParser()
    parser.add_argument("-f", "--file")
    parser.add_argument("--nprocs", "-n", type=int, default=1)

    args = parser.parse_args()
    # mols = [f for f in os.listdir(args.directory) if f.endswith(".xyz")]


    mols = read(args.file, index=":")

    print(f"Extracting atomtypes for {len(mols)} configs")


    # write mol out to mol2, do the conversion, also convert to ase atoms

    # old_dir = os.getcwd()
    # for idx, mol in tqdm(enumerate(mols)):
    values = [(idx, mol) for (idx, mol) in enumerate(mols)]
    # extract_atomtypes(values[0])

    p = Pool(args.nprocs)

    configs = p.map(extract_atomtypes, values)
    configs = [c for c in configs if c is not None]
    print(len(configs))
        

    with open("mols.xyz", 'w') as f:
        write(f, configs)

        # now we have atomtypes for thing in the mol, just append them as an extra data field to the atoms object




if __name__ == "__main__":
    main()
