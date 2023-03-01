# Command line script to extract openFF nonbonded forces and atom types from atoms and append to a extxyz file

from ase.io import read, write
import os
from openff.toolkit.topology import Molecule
from openmm.unit import (
    kelvin,
    picosecond,
    femtosecond,
    nanometer,
    angstrom,
    kilojoule_per_mole,
)
from shutil import rmtree
from openmm import LangevinMiddleIntegrator
import numpy as np
from openmm.app import (
    Simulation,
    Element,
    Topology,
    StateDataReporter,
    PDBReporter,
    DCDReporter,
    ForceField,
    PDBFile,
    Modeller,
    PME,
)
import subprocess
from time import time
from rdkit import Chem
from argparse import ArgumentParser
import openmm
from openff.toolkit.topology import Topology as OFFTopology
from typing import Iterable
from openmmtools.openmm_torch.utils import (
    remove_bonded_forces,
    initialize_mm_forcefield,
    set_smff,
)
from multiprocessing import Pool
from ase.units import eV, kJ, mol
import tempfile
from espaloma_charge.openff_wrapper import EspalomaChargeToolkitWrapper
from openmm.unit import elementary_charge

toolkit_registry = EspalomaChargeToolkitWrapper()


def extract_nonbonded_components(values):
    t1 = time() 
    (atoms, smff) = values
    # takes an ase atoms object and a smiles string, moves nonbonded components of the forcefield to a new forcegroup, runs a single step of the integrator, attaches np array of nb_forces to the atoms object
    # parsed_smiles = f.readlines()
    old_dir = os.getcwd()
    try:
       
        tmpdir = tempfile.mkdtemp()


        with open(os.path.join(tmpdir, "mol.xyz"), "w") as f:
            write(f, atoms)
            print(tmpdir)

        cmd = f"python $XYZ2MOL {os.path.join(tmpdir, 'mol.xyz')} -o sdf > {os.path.join(tmpdir, 'mol.sdf')}"
        os.system(cmd)
        subprocess.run(cmd, shell=True, check=True, capture_output=True, timeout=60)
        


        cmd = f"obabel -isdf {os.path.join(tmpdir, 'mol.sdf')} -omol2 -O {os.path.join(tmpdir, 'mol.mol2')}"
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        
        os.chdir(tmpdir)
        # cmd = f"antechamber -nc {formal_charge} -fi mol2 -i {os.path.join(tmpdir, 'mol.mol2')} -c gas -fo mol2 -o {os.path.join(tmpdir, 'mol_processed.mol2')} -dr n"
        cmd = f"atomtype -i mol.mol2 -f mol2 -o mol.ac -p gaff2"
        output = subprocess.run(
            cmd, capture_output=True, universal_newlines=True, shell=True
        )
        if output.returncode != 0:
            print(output.stderr)
            raise Exception("Error in antechamber")
        
        # extract the atom types and partial charges
        cmd = "cat  mol.ac | grep UNL | awk '{ print $10 } ' > atomtypes.txt"
        subprocess.run(
            cmd, capture_output=True, universal_newlines=True, check=True, shell=True
        )

        with open("atomtypes.txt", "r") as f:
            # it captures an extra line at the bottom of the mol2 that we don't need
            atomtypes = [l.strip() for l in f.readlines()]
       
        os.chdir(old_dir)
        atoms.set_array("atomtypes", np.array(atomtypes))

        molecule = Molecule.from_file(os.path.join(tmpdir, "mol.sdf"), allow_undefined_stereo=True)
        topology = molecule.to_topology().to_openmm()
        molecule.compute_partial_charges_am1bcc()


        partials = molecule.partial_charges.value_in_unit(elementary_charge)
        atoms.set_array("partial_charges", np.array(partials))


        smff = set_smff(smff)

        forcefield = initialize_mm_forcefield(molecule=molecule, smff=smff)


        system = forcefield.createSystem(
            topology=molecule.to_topology().to_openmm(),
            constraints=None,
        )

        atoms_idx = [i for i in range(molecule.n_atoms)]
        # remove all bonded forces from the system, leave only the NB forces
        nonbonded_system = remove_bonded_forces(
            system, atoms=atoms_idx, removeInSet=True, removeConstraints=True
        )
       
        # step an integrator
        temperature = 298.15 * kelvin
        frictionCoeff = 1 / picosecond
        timeStep = 1 * femtosecond
        integrator = LangevinMiddleIntegrator(temperature, frictionCoeff, timeStep)
        simulation = Simulation(topology, nonbonded_system, integrator)
        simulation.context.setPositions(atoms.get_positions() / 10)
        # convert to eV and eV/A
        state = simulation.context.getState(getForces=True, getEnergy=True)
        forces = (
            state.getForces(asNumpy=True).value_in_unit(kilojoule_per_mole / angstrom)
            * kJ
            / (mol * eV)
        )
        energy = (
            state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
            * kJ
            / (mol * eV)
        )
        sr_energy = atoms.info["energy"] - energy
        sr_forces = atoms.arrays["forces"] - forces
        atoms.new_array("nb_forces", forces)
        atoms.info["nb_energy"] = energy
        atoms.new_array("sr_forces", sr_forces)
        atoms.info["sr_energy"] = sr_energy
        rmtree(tmpdir)
        print(f"Ran calculation in {time() - t1} seconds")
        return atoms

    except Exception as e:
        print("Error!!!")
        print(e)
        rmtree(tmpdir)
        return



def main():
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", help="path to xyz file containing configs")
    parser.add_argument("-n", "--nprocs", type=int)
    parser.add_argument("-o", "--output", help="path to output xyz file")
    parser.add_argument("-r", "--range", help="range of configs to extract")
    parser.add_argument(
        "-smff", help="which version of the OFF forcefield to use", default="1.0"
    )
    args = parser.parse_args()
    p = Pool(args.nprocs)

    configs = read(args.file, args.range)
    print(f"Evaluating {len(configs)} configs", flush=True)
    values = [(conf, args.smff) for conf in configs]

    configs = p.map(extract_nonbonded_components, values)
    configs = [c for c in configs if c is not None]

    with open(args.output, "w") as f:
        write(f, configs)


if __name__ == "__main__":
    main()
