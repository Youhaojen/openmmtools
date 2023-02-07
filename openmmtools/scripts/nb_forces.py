# Command line script to extract openFF nonbonded forces and atom types from atoms and append to a extxyz file

from ase.io import read, write
from openff.toolkit.topology import Molecule
from openmm.unit import kelvin, picosecond, femtosecond, nanometer
from openmm import LangevinMiddleIntegrator
from openmm.app import (
    Simulation,
    StateDataReporter,
    PDBReporter,
    DCDReporter,
    ForceField,
    PDBFile,
    Modeller,
    PME,
)
from argparse import ArgumentParser
import openmm
from typing import Iterable
from openmmtools.openmm_torch.utils import (
    remove_bonded_forces,
    initialize_mm_forcefield,
    set_smff,
)


def extract_nonbonded_components(path: str, smiles: str, smff: str):
    # takes an ase atoms object and a smiles string, moves nonbonded components of the forcefield to a new forcegroup, runs a single step of the integrator, attaches np array of nb_forces to the atoms object
    configs = read(path, ":")
    with open(smiles, "r") as f:
        parsed_smiles = f.readlines()
    for atoms, smile in zip(configs, parsed_smiles):
        smile = smile.split()[0]
        box_vectors = atoms.get_cell() / 10
        print(box_vectors)
        molecule = Molecule.from_smiles(smile, hydrogens_are_explicit=False)
        topology = molecule.to_topology().to_openmm()
        if max(atoms.get_cell().cellpar()[:3]) > 0:
            topology.setPeriodicBoxVectors(vectors=box_vectors)

        smff = set_smff(smff)

        forcefield = initialize_mm_forcefield(molecule=molecule, smff=smff)

        system = forcefield.createSystem(
            topology=topology,
            nonbondedMethod=PME,
            nonbondedCutoff=1.0 * nanometer,
            constraints=None,
        )
        atoms_idx = [atom.index for atom in topology.atoms()]
        # atoms = get_atoms_from_resname(topology=topology, resname="MOL")
        system = remove_bonded_forces(
            system, atoms=atoms_idx, removeInSet=True, removeConstraints=False
        )

        # step an integrator
        temperature = 298.15 * kelvin
        frictionCoeff = 1 / picosecond
        timeStep = 1 * femtosecond
        integrator = LangevinMiddleIntegrator(temperature, frictionCoeff, timeStep)
        simulation = Simulation(topology, system, integrator)
        simulation.context.setPositions(atoms.get_positions() / 10)
        state = simulation.context.getState(getForces=True)
        forces = state.getForces(asNumpy=True)
        print(forces)
        atoms.new_array("nb_forces", forces)

    return configs


def main():
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", help="path to xyz file containing configs")
    parser.add_argument("-s", "--smiles", help="path to xyz file containing configs")
    parser.add_argument(
        "-smff", help="which version of the OFF forcefield to use", default="1.0"
    )
    args = parser.parse_args()

    configs = extract_nonbonded_components(
        path=args.file, smiles=args.smiles, smff=args.smff
    )

    with open("output_configs.extxyz", "w") as f:
        write(f, configs)


if __name__ == "__main__":
    main()
