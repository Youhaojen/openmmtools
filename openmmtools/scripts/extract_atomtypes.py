from argparse import ArgumentParser
import os
import subprocess
from rdkit.Chem.rdmolfiles import MolToXYZFile, MolToMolBlock
from ase.io import read, write
import numpy as np

def main():
    parser = ArgumentParser()
    parser.add_argument("-d", "--directory")

    args = parser.parse_args()
    mols = [f for f in os.listdir(args.directory) if f.endswith(".xyz")]

    print(mols)

    configs = []

    # write mol out to mol2, do the conversion, also convert to ase atoms

    old_dir = os.getcwd()
    for mol in mols:
        output_dir = os.path.join(args.directory, f"{mol.split('.')[0]}_processing")
        # write out mol2 file
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)
        # convert mol to mol2
        cmd = f"obabel -ixyz ../{mol} -omol2 -O {mol.split('.')[0]}.mol2"
        subprocess.run(cmd, shell=True, check=True)
        cmd = (
            f"antechamber -fi mol2 -i {mol.split('.')[0]}.mol2 -c gas -fo mol2 -o mol_processed.mol2"
        )
        output = subprocess.run(
            cmd, capture_output=True, universal_newlines=True, check=True, shell=True
        )
        for line in output.stdout.splitlines():
            print(line)
        cmd = "cat mol_processed.mol2 | grep UNL | awk '{ print $6 } ' > atomtypes.txt"
        subprocess.run(cmd, capture_output=True, universal_newlines=True, check=True, shell=True)
        with open("atomtypes.txt", "r") as f:
            # it captures an extra line at the bottom of the mol2 that we don't need
            atomtypes = [l.strip() for l in f.readlines()[:-1]]
        print(atomtypes)
        # open the resulting file, strip the
        atoms = read(f"../{mol}")
        atoms.set_array("atomtypes", np.array(atomtypes))
        configs.append(atoms)
        os.chdir(old_dir)

    with open("mols.xyz", 'w') as f:
        write(f, configs)

        # now we have atomtypes for thing in the mol, just append them as an extra data field to the atoms object




if __name__ == "__main__":
    main()
