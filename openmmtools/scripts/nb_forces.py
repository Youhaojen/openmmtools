# Command line script to extract openFF nonbonded forces and atom types from atoms and append to a extxyz file

from ase.io import read, write
from openff.toolkit.topology import Molecule
from openmm.unit import kelvin, picosecond, femtosecond, nanometer, angstrom, kilojoule_per_mole
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
from multiprocessing import Pool
from ase.units import eV, kJ, mol



def extract_nonbonded_components(values):
    (atoms, smff) = values
    # takes an ase atoms object and a smiles string, moves nonbonded components of the forcefield to a new forcegroup, runs a single step of the integrator, attaches np array of nb_forces to the atoms object
    # parsed_smiles = f.readlines()
    smile = atoms.info['smiles']
    smile = smile.split()[0]
    print(smile)
    if atoms.get_cell() is not None:

        atoms.set_cell([50,50,50])
    box_vectors = atoms.get_cell() / 10
    try:
        molecule = Molecule.from_smiles(smile, hydrogens_are_explicit=False, allow_undefined_stereo=True)
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
        bonded_system =  remove_bonded_forces(
            system, atoms=atoms_idx, removeInSet=False, removeConstraints=False
        )

        # step an integrator
        temperature = 298.15 * kelvin
        frictionCoeff = 1 / picosecond
        timeStep = 1 * femtosecond
        integrator = LangevinMiddleIntegrator(temperature, frictionCoeff, timeStep)
        simulation = Simulation(topology, system, integrator)
        simulation.context.setPositions(atoms.get_positions() / 10)

        # convert to eV and eV/A
        state = simulation.context.getState(getForces=True, getEnergy=True)
        forces = state.getForces(asNumpy=True).value_in_unit(kilojoule_per_mole / angstrom)  * kJ / (mol * eV)
        energy = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole) * kJ / (mol * eV) 
        print(energy)
        sr_energy = atoms.info["energy"] - energy
        sr_forces = atoms.arrays["forces"] - forces
        print(forces)
        atoms.new_array("nb_forces", forces)
        atoms.info["nb_energy"]= energy
        atoms.new_array("sr_forces", sr_forces)
        atoms.info["sr_energy"]= sr_energy

    except Exception as e:
        print("Error!!!")
        print(e)

    return atoms


def main():
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", help="path to xyz file containing configs")
    parser.add_argument("-n", "--nprocs", type=int)
    parser.add_argument(
        "-smff", help="which version of the OFF forcefield to use", default="1.0"
    )
    args = parser.parse_args()
    p = Pool(args.nprocs)

    configs = read(args.file, ":8")
    values = [(conf, args.smff) for conf in configs]

    configs = p.map(extract_nonbonded_components, values)

    with open("output_configs.extxyz", "w") as f:
        write(f, configs)


if __name__ == "__main__":
    main()
