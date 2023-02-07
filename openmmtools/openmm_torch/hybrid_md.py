import sys
import mdtraj
from ase.io import read
import torch
import time
import numpy as np
import sys
from ase import Atoms
from rdkit.Chem.rdmolfiles import MolFromPDBFile, MolFromXYZFile
from openmm.openmm import System
from typing import List, Tuple, Optional, Union, Type
from openmm import (
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
    CustomTorsionForce,
)
import matplotlib.pyplot as plt
from openmmtools.integrators import AlchemicalNonequilibriumLangevinIntegrator
from mdtraj.reporters import HDF5Reporter
from mdtraj.geometry.dihedral import indices_phi, indices_psi
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
from openmm.app.metadynamics import Metadynamics, BiasVariable
import openmm
from typing import Iterable
from openmm.app.topology import Topology
from openmm.app.element import Element
from openmm.unit import (
    kelvin,
    picosecond,
    femtosecond,
    kilojoule_per_mole,
    picoseconds,
    femtoseconds,
    bar,
    nanometer,
    nanometers,
    molar,
    angstrom,
)
from openff.toolkit.topology import Molecule


from openmmml.models.mace_potential import MacePotentialImplFactory
from openmmml.models.anipotential import ANIPotentialImplFactory
from openmmml import MLPotential

from openmmtools.openmm_torch.repex import (
    MixedSystemConstructor,
    RepexConstructor,
    get_atoms_from_resname,
)
from openmmtools.openmm_torch.utils import (
    initialize_mm_forcefield,
    set_smff,
)
from tempfile import mkstemp
import os
import logging


def get_xyz_from_mol(mol):

    xyz = np.zeros((mol.GetNumAtoms(), 3))
    conf = mol.GetConformer()
    for i in range(conf.GetNumAtoms()):
        position = conf.GetAtomPosition(i)
        xyz[i, 0] = position.x
        xyz[i, 1] = position.y
        xyz[i, 2] = position.z
    return xyz


MLPotential.registerImplFactory("mace", MacePotentialImplFactory())
MLPotential.registerImplFactory("ani2x", ANIPotentialImplFactory())


# platform = Platform.getPlatformByName("CUDA")
# platform.setPropertyDefaultValue("DeterministicForces", "true")

logger = logging.getLogger("INFO")


class MixedSystem:
    forcefields: List[str]
    padding: float
    ionicStrength: float
    nonbondedCutoff: float
    resname: str
    nnpify_type: str
    potential: str
    temperature: float
    friction_coeff: float
    timestep: float
    dtype: torch.dtype
    output_dir: str
    neighbour_list: str
    openmm_precision: str
    SM_FF: str
    modeller: Modeller
    mixed_system: System
    decoupled_system: Optional[System]

    def __init__(
        self,
        file: str,
        ml_mol: str,
        model_path: str,
        forcefields: List[str],
        resname: str,
        nnpify_type: str,
        padding: float,
        ionicStrength: float,
        nonbondedCutoff: float,
        potential: str,
        temperature: float,
        dtype: torch.dtype,
        neighbour_list: str,
        output_dir: str,
        system_type: str,
        boxvecs: Optional[List[List]] = None,
        friction_coeff: float = 1.0,
        timestep: float = 1.0,
        smff: str = "1.0",
        pressure: Optional[float] = None,
        cv1: Optional[str] = None,
        cv2: Optional[str] = None,
    ) -> None:

        self.forcefields = forcefields
        self.padding = padding
        self.ionicStrength = ionicStrength
        self.nonbondedCutoff = nonbondedCutoff
        self.resname = resname
        self.nnpify_type = nnpify_type
        self.potential = potential
        self.temperature = temperature
        self.friction_coeff = friction_coeff / picosecond
        self.timestep = timestep * femtosecond
        self.dtype = dtype
        self.cv1 = cv1
        self.cv2 = cv2
        self.output_dir = output_dir
        self.neighbour_list = neighbour_list
        self.openmm_precision = "Double" if dtype == torch.float64 else "Mixed"
        self.boxvecs = (
            boxvecs if boxvecs is not None else [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]
        )
        logger.debug(f"OpenMM will use {self.openmm_precision} precision")

        self.SM_FF = set_smff(smff)
        logger.info(f"Using SMFF {self.SM_FF}")

        os.makedirs(self.output_dir, exist_ok=True)
        if system_type == "pure":
            print("Creating pure system")
            self.create_pure_system(
                file=file,
                model_path=model_path,
                pressure=pressure,
            )
        else:
            self.create_mixed_system(
                file=file,
                ml_mol=ml_mol,
                model_path=model_path,
                system_type=system_type,
                pressure=pressure,
            )

    def initialize_ase_atoms(self, ml_mol: str) -> Tuple[Atoms, Molecule]:
        """Generate the ase atoms object from the

        :param str ml_mol: file path or smiles
        :return Tuple[Atoms, Molecule]: ase Atoms object and initialised openFF molecule
        """
        # ml_mol can be a path to a file, or a smiles string
        if os.path.isfile(ml_mol):
            if ml_mol.endswith(".pdb"):
                # openFF refuses to work with pdb or xyz files, rely on rdkit to do the convertion to a mol first
                molecule = MolFromPDBFile(ml_mol)
                logger.warning(
                    "Initializing topology from pdb - this can lead to valence errors, check your starting structure carefully!"
                )
                molecule = Molecule.from_rdkit(
                    molecule, hydrogens_are_explicit=True, allow_undefined_stereo=True
                )
            elif ml_mol.endswith(".xyz"):
                molecule = MolFromXYZFile(ml_mol)
                molecule = Molecule.from_rdkit(molecule, hydrogens_are_explicit=True)
            else:
                # assume openFF will handle the format otherwise
                molecule = Molecule.from_file(ml_mol, allow_undefined_stereo=True)
        else:
            try:
                molecule = Molecule.from_smiles(ml_mol)
            except:
                raise ValueError(
                    f"Attempted to interpret arg {ml_mol} as a SMILES string, but could not parse"
                )

        _, tmpfile = mkstemp(suffix=".xyz")
        molecule._to_xyz_file(tmpfile)
        print(tmpfile)
        atoms = read(tmpfile)
        # os.remove(tmpfile)
        return atoms, molecule

    def create_pure_system(
        self,
        file: str,
        model_path: str,
        pressure: Union[float, Type[None]],
    ) -> None:
        """Creates the mixed system from a purely mm system

        :param str file: input pdb file
        :param str model_path: path to the mace model
        :return Tuple[System, Modeller]: return mixed system and the modeller for topology + position access by downstream methods
        """
        # initialize the ase atoms for MACE
        atoms = read(file)
        if file.endswith(".xyz"):
            pos = atoms.get_positions() / 10
            box_vectors = atoms.get_cell() / 10
            print("Got box vectors", box_vectors)
            elements = atoms.get_chemical_symbols()

            # Create a topology object
            topology = Topology()

            # Add atoms to the topology
            chain = topology.addChain()
            res = topology.addResidue("mace_system", chain)
            for i, (element, position) in enumerate(zip(elements, pos)):
                e = Element.getBySymbol(element)
                topology.addAtom(str(i), e, res)
            # if there is a periodic box specified add it to the Topology
            if max(atoms.get_cell().cellpar()[:3]) > 0:
                print("Adding periodic box to topology ...")
                topology.setPeriodicBoxVectors(vectors=box_vectors)

            print(f"Initialized topology with {pos.shape} positions")

            self.modeller = Modeller(topology, pos)

        elif file.endswith(".sdf"):
            molecule = Molecule.from_file(file)
            # input_file = molecule
            topology = molecule.to_topology().to_openmm()
            # Hold positions in nanometers
            positions = get_xyz_from_mol(molecule.to_rdkit()) / 10

            print(f"Initialized topology with {positions.shape} positions")

            self.modeller = Modeller(topology, positions)
        else:
            raise NotImplementedError
        pbc = True if topology.getPeriodicBoxVectors() is not None else False
        print("system uses pbc: ", pbc)
        ml_potential = MLPotential("mace")
        self.mixed_system = ml_potential.createSystem(
            topology,
            atoms_obj=atoms,
            filename=model_path,
            dtype=self.dtype,
            nl=self.neighbour_list,
            pbc=pbc,
        )

        if pressure is not None:
            print(f"Pressure will be maintained at {pressure} bar with MC barostat")
            barostat = MonteCarloBarostat(pressure * bar, self.temperature * kelvin)
            barostat.setFrequency(25)  # 25 timestep is the default
            self.mixed_system.addForce(barostat)

    def create_mixed_system(
        self,
        file: str,
        model_path: str,
        ml_mol: str,
        system_type: str,
        pressure: Optional[float],
    ) -> None:
        """Creates the mixed system from a purely mm system

        :param str file: input pdb file
        :param str smiles: smiles of the small molecule, only required when passed as part of the complex
        :param str model_path: path to the mace model
        :return Tuple[System, Modeller]: return mixed system and the modeller for topology + position access by downstream methods
        """
        # initialize the ase atoms for MACE

        atoms, molecule = self.initialize_ase_atoms(ml_mol)
        # set the default topology to that of the ml molecule, this will get overwritten below
        # topology = molecule.to_topology().to_openmm()

        # Handle a complex, passed as a pdb file
        if file.endswith(".pdb"):
            input_file = PDBFile(file)
            topology = input_file.getTopology()

            # if pure_ml_system specified, we just need to parse the input file
            # if not pure_ml_system:
            self.modeller = Modeller(input_file.topology, input_file.positions)
            print(f"Initialized topology with {len(input_file.positions)} positions")

        # Handle a small molecule/small periodic system, passed as an sdf or xyz
        elif file.endswith(".sdf") or file.endswith(".xyz"):
            # this is unnecessary, we have run exactly the same thing above
            # molecule = Molecule.from_file(file)
            input_file = molecule
            topology = molecule.to_topology().to_openmm()
            # Hold positions in nanometers
            positions = get_xyz_from_mol(molecule.to_rdkit()) / 10

            print(f"Initialized topology with {positions.shape} positions")

            self.modeller = Modeller(topology, positions)

        if system_type == "pure":
            print("boxvecs", self.boxvecs)
            # we have the input_file, create the system directly from the mace potential
            # TODO: add a function to compute periodic box vectors to enforce a minimum padding distance to each box wall
            atoms.set_cell(np.array(self.boxvecs) * 10)  # set in angstroms
            # atoms.set_cell([50,50,50])
            topology.setPeriodicBoxVectors(self.boxvecs)
            ml_potential = MLPotential("mace")
            self.mixed_system = ml_potential.createSystem(
                topology,
                atoms_obj=atoms,
                filename=model_path,
                dtype=self.dtype,
                nl=self.neighbour_list,
            )

        # Handle the mixed systems with a classical forcefield
        elif system_type in ["hybrid", "decoupled"]:
            forcefield = initialize_mm_forcefield(
                molecule=molecule, forcefields=self.forcefields, smff=self.SM_FF
            )
            self.modeller.addSolvent(
                forcefield,
                padding=self.padding * nanometers,
                ionicStrength=self.ionicStrength * molar,
                neutralize=True,
            )

            omm_box_vecs = self.modeller.topology.getPeriodicBoxVectors()
            # print(omm_box_vecs)
            atoms.set_cell(
                [
                    omm_box_vecs[0][0].value_in_unit(angstrom),
                    omm_box_vecs[1][1].value_in_unit(angstrom),
                    omm_box_vecs[2][2].value_in_unit(angstrom),
                ]
            )

            system = forcefield.createSystem(
                self.modeller.topology,
                nonbondedMethod=PME,
                nonbondedCutoff=self.nonbondedCutoff * nanometer,
                constraints=None,
            )
            if pressure is not None:
                logger.info(
                    f"Pressure will be maintained at {pressure} bar with MC barostat"
                )
                system.addForce(
                    MonteCarloBarostat(pressure * bar, self.temperature * kelvin)
                )

            # write the final prepared system to disk
            with open(os.path.join(self.output_dir, "prepared_system.pdb"), "w") as f:
                PDBFile.writeFile(
                    self.modeller.topology, self.modeller.getPositions(), file=f
                )
            if system_type == "hybrid":
                logger.debug("Creating hybrid system")
                self.mixed_system = MixedSystemConstructor(
                    system=system,
                    topology=self.modeller.topology,
                    nnpify_id=self.resname,
                    nnp_potential=self.potential,
                    nnpify_type=self.nnpify_type,
                    atoms_obj=atoms,
                    filename=model_path,
                    dtype=self.dtype,
                    nl=self.neighbour_list,
                    pbc=True if system.usesPeriodicBoundaryConditions() else False,
                ).mixed_system

            # optionally, add the alchemical customCVForce for the nonbonded interactions to run ABFE edges
            # else:
            #     logger.debug("Creating decoupled system")
            #     self.mixed_system = MixedSystemConstructor(
            #         system=system,
            #         topology=self.modeller.topology,
            #         nnpify_resname=self.resname,
            #         nnp_potential=self.potential,
            #         atoms_obj=atoms,
            #         filename=model_path,
            #         dtype=self.dtype,
            #         nl=self.neighbour_list,
            #     ).decoupled_system

        else:
            raise ValueError(f"system type {system_type} not recognised - aborting!")

    def run_metadynamics(
        # self, topology: Topology, cv1_dsl_string: str, cv2_dsl_string: str
        self,
        topology: Topology,
    ) -> Metadynamics:
        # run well-tempered metadynamics
        mdtraj_topology = mdtraj.Topology.from_openmm(topology)

        cv1_atom_indices = indices_psi(mdtraj_topology)
        cv2_atom_indices = indices_phi(mdtraj_topology)
        print("cv1_atom_indices", cv1_atom_indices)
        # logger.info(f"Selcted cv1 torsion atoms {cv1_atom_indices}")
        # cv2_atom_indices = mdtraj_topology.select(cv2_dsl_string)
        # logger.info(f"Selcted cv2 torsion atoms {cv2_atom_indices}")
        # takes the mixed system parametrised in the init method and performs metadynamics
        # in the canonical case, this should just use the psi-phi backbone angles of the peptide

        cv1 = CustomTorsionForce("theta")
        # cv1.addTorsion(cv1_atom_indices)
        cv1.addTorsion(cv1_atom_indices)
        phi = BiasVariable(cv1, -np.pi, np.pi, biasWidth=0.5, periodic=True)

        cv2 = CustomTorsionForce("theta")
        cv2.addTorsion(cv2_atom_indices)
        psi = BiasVariable(cv2, -np.pi, np.pi, biasWidth=0.5, periodic=True)
        os.makedirs(os.path.join(self.output_dir, "metaD"), exist_ok=True)
        meta = Metadynamics(
            self.mixed_system,
            [psi, phi],
            temperature=self.temperature,
            biasFactor=10.0,
            height=1.0 * kilojoule_per_mole,
            frequency=100,
            biasDir=os.path.join(self.output_dir, "metaD"),
            saveFrequency=100,
        )

        return meta

    def run_mixed_md(
        self, steps: int, interval: int, output_file: str, run_metadynamics: bool
    ):
        """Runs plain MD on the mixed system, writes a pdb trajectory

        :param int steps: number of steps to run the simulation for
        :param int interval: reportInterval attached to reporters
        """
        integrator = LangevinMiddleIntegrator(
            self.temperature, self.friction_coeff, self.timestep
        )

        if run_metadynamics:
            # TODO: this should handle creating the customCVs for us from atom selection or something
            meta = self.run_metadynamics(
                topology=self.modeller.topology
                # cv1_dsl_string=self.cv1_dsl_string, cv2_dsl_string=self.cv2_dsl_string
            )

        logger.debug(f"Running mixed MD for {steps} steps")
        simulation = Simulation(
            self.modeller.topology,
            self.mixed_system,
            integrator,
            platformProperties={"Precision": self.openmm_precision},
        )
        simulation.context.setPositions(self.modeller.getPositions())
        logging.info("Minimising energy...")
        simulation.minimizeEnergy()
        minimised_state = simulation.context.getState(
            getPositions=True, getVelocities=True, getForces=True
        )
        with open(os.path.join(self.output_dir, f"minimised_system.pdb"), "w") as f:
            PDBFile.writeFile(
                self.modeller.topology, minimised_state.getPositions(), file=f
            )

        reporter = StateDataReporter(
            file=sys.stdout,
            reportInterval=interval,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            speed=True,
        )
        simulation.reporters.append(reporter)
        simulation.reporters.append(
            PDBReporter(
                file=os.path.join(self.output_dir, output_file),
                reportInterval=interval,
                enforcePeriodicBox=True,
            )
        )
        dcd_reporter = DCDReporter(
            file=os.path.join(self.output_dir, "output.dcd"), reportInterval=interval
        )
        simulation.reporters.append(dcd_reporter)
        hdf5_reporter = HDF5Reporter(
            file=os.path.join(self.output_dir, output_file[:-4] + ".h5"),
            reportInterval=interval,
            velocities=True,
        )
        simulation.reporters.append(hdf5_reporter)

        if run_metadynamics:
            logger.info("Running metadynamics")
            # handles running the simulation with metadynamics
            meta.step(simulation, steps)

            fe = meta.getFreeEnergy()
            print(fe)
            fig, ax = plt.subplots(1, 1)
            ax.imshow(fe)
            fig.savefig(os.path.join(self.output_dir, "free_energy.png"))

        else:
            simulation.step(steps)

    def run_repex(
        self,
        replicas: int,
        restart: bool,
        steps: int,
        intervals_per_lambda_window: int = 10,
        steps_per_equilibration_interval: int = 100,
        equilibration_protocol: str = "minimise",
    ) -> None:
        repex_file_exists = os.path.isfile(os.path.join(self.output_dir, "repex.nc"))
        # even if restart has been set, disable if the checkpoint file was not found, enforce minimising the system
        if not repex_file_exists:
            restart = False
        sampler = RepexConstructor(
            mixed_system=self.mixed_system,
            initial_positions=self.modeller.getPositions(),
            intervals_per_lambda_window=2 * replicas,
            steps_per_equilibration_interval=steps_per_equilibration_interval,
            equilibration_protocol=equilibration_protocol,
            # repex_storage_file="./out_complex.nc",
            temperature=self.temperature * kelvin,
            n_states=replicas,
            restart=restart,
            mcmc_moves_kwargs={
                "timestep": 1.0 * femtoseconds,
                "collision_rate": 1.0 / picoseconds,
                "n_steps": 1000,
                "reassign_velocities": False,
                "n_restart_attempts": 20,
            },
            replica_exchange_sampler_kwargs={
                "number_of_iterations": steps,
                "online_analysis_interval": 10,
                "online_analysis_minimum_iterations": 10,
            },
            storage_kwargs={
                "storage": os.path.join(self.output_dir, "repex.nc"),
                "checkpoint_interval": 1,
                "analysis_particle_indices": get_atoms_from_resname(
                    topology=self.modeller.topology,
                    nnpify_id=self.resname,
                    nnpify_type=self.nnpify_type,
                ),
            },
        ).sampler

        # do not minimsie if we are hot-starting the simulation from a checkpoint
        if not restart and equilibration_protocol == "minimise":
            logging.info("Minimizing system...")
            t1 = time.time()
            sampler.minimize()

            logging.info(f"Minimised system  in {time.time() - t1} seconds")
            # we want to write out the positions after the minimisation - possibly something weird is going wrong here and it's ending up in a weird conformation

        sampler.run()

    def run_neq_switching(self, steps: int, interval: int) -> float:
        """Compute the protocol work performed by switching from the MM description to the MM/ML through lambda_interpolate

        :param int steps: number of steps in non-equilibrium switching simulation
        :param int interval: reporterInterval
        :return float: protocol work from the integrator
        """
        alchemical_functions = {"lambda_interpolate": "lambda"}
        integrator = AlchemicalNonequilibriumLangevinIntegrator(
            alchemical_functions=alchemical_functions,
            nsteps_neq=steps,
            temperature=self.temperature,
            collision_rate=self.friction_coeff,
            timestep=self.timestep,
        )

        simulation = Simulation(
            self.modeller.topology,
            self.mixed_system,
            integrator,
            platformProperties={"Precision": "Double", "Threads": 16},
        )
        simulation.context.setPositions(self.modeller.getPositions())

        logging.info("Minimising energy")
        simulation.minimizeEnergy()

        reporter = StateDataReporter(
            file=sys.stdout,
            reportInterval=interval,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            speed=True,
            totalSteps=steps,
            remainingTime=True,
        )
        simulation.reporters.append(reporter)
        # Append the snapshots to the pdb file
        simulation.reporters.append(
            PDBReporter(
                os.path.join(self.output_dir, "output_frames.pdb"),
                steps / 80,
                enforcePeriodicBox=True,
            )
        )
        # We need to take the final state
        simulation.step(steps)
        protocol_work = (integrator.get_protocol_work(dimensionless=True),)
        return protocol_work


# class MACESystem:
#     potential: str
#     temperature: float
#     friction_coeff: float
#     timestep: float
#     dtype: torch.dtype
#     output_dir: str
#     neighbour_list: str
#     openmm_precision: str
#     SM_FF: str
#     modeller: Modeller

#     def __init__(
#         self,
#         file: str,
#         model_path: str,
#         potential: str,
#         output_dir: str,
#         temperature: float,
#         pressure: Optional[float] = None,
#         dtype: torch.dtype = torch.float64,
#         neighbour_list: str = "torch_nl_n2",
#         friction_coeff: float = 1.0,
#         timestep: float = 1.0,
#         smff: str = "1.0",
#     ) -> None:

#         self.potential = potential
#         self.temperature = temperature
#         self.friction_coeff = friction_coeff / picosecond
#         self.timestep = timestep * femtosecond
#         self.dtype = dtype
#         self.output_dir = output_dir
#         self.neighbour_list = neighbour_list
#         self.openmm_precision = "Double" if dtype == torch.float64 else "Mixed"
#         logger.debug(f"OpenMM will use {self.openmm_precision} precision")

#         if smff == "1.0":
#             self.SM_FF = "openff_unconstrained-1.0.0.offxml"
#             logger.info("Using openff-1.0 unconstrained forcefield")
#         elif smff == "2.0":
#             self.SM_FF = "openff_unconstrained-2.0.0.offxml"
#             logger.info("Using openff-2.0 unconstrained forcefield")
#         else:
#             raise ValueError(f"Small molecule forcefield {smff} not recognised")

#         os.makedirs(self.output_dir, exist_ok=True)

#         self.create_system(file=file, model_path=model_path, pressure=pressure)

#     def create_system(
#         self,
#         file: str,
#         model_path: str,
#         pressure: Union[float, Type[None]],
#     ) -> None:
#         """Creates the mixed system from a purely mm system

#         :param str file: input pdb file
#         :param str model_path: path to the mace model
#         :return Tuple[System, Modeller]: return mixed system and the modeller for topology + position access by downstream methods
#         """
#         # initialize the ase atoms for MACE
#         atoms = read(file)
#         if file.endswith(".xyz"):
#             pos = atoms.get_positions() / 10
#             box_vectors = atoms.get_cell() / 10
#             elements = atoms.get_chemical_symbols()

#             # Create a topology object
#             topology = Topology()

#             # Add atoms to the topology
#             chain = topology.addChain()
#             res = topology.addResidue("mace_system", chain)
#             for i, (element, position) in enumerate(zip(elements, pos)):
#                 e = Element.getBySymbol(element)
#                 topology.addAtom(str(i), e, res)
#             # if there is a periodic box specified add it to the Topology
#             if max(atoms.get_cell().cellpar()[:3]) > 0:
#                 topology.setPeriodicBoxVectors(vectors=box_vectors)

#             print(f"Initialized topology with {pos.shape} positions")

#             self.modeller = Modeller(topology, pos)
#         # Handle a system, passed as a pdb file
#         # if file.endswith(".pdb"):
#         #     input_file = PDBFile(file)
#         #     topology = input_file.getTopology()

#         #     # if pure_ml_system specified, we just need to parse the input file
#         #     # if not pure_ml_system:
#         #     self.modeller = Modeller(input_file.topology, input_file.positions)
#         #     print(f"Initialized topology with {len(input_file.positions)} positions")

#         elif file.endswith(".sdf"):
#             molecule = Molecule.from_file(file)
#             # input_file = molecule
#             topology = molecule.to_topology().to_openmm()
#             # Hold positions in nanometers
#             positions = get_xyz_from_mol(molecule.to_rdkit()) / 10

#             print(f"Initialized topology with {positions.shape} positions")

#             self.modeller = Modeller(topology, positions)
#         else:
#             raise NotImplementedError

#         ml_potential = MLPotential("mace")
#         self.mixed_system = ml_potential.createSystem(
#             topology, atoms_obj=atoms, filename=model_path, dtype=self.dtype
#         )

#         if pressure is not None:
#             print(f"Pressure will be maintained at {pressure} bar with MC barostat")
#             barostat = MonteCarloBarostat(pressure * bar, self.temperature * kelvin)
#             # barostat.setFrequency(25)  25 timestep is the default
#             self.mixed_system.addForce(barostat)

#     def run_md(self, steps: int, interval: int, output_file: str) -> float:
#         """Runs Langevin MD with MACE, writes a pdb trajectory

#         :param int steps: number of steps to run the simulation for
#         :param int interval: reportInterval attached to reporters
#         """
#         integrator = LangevinMiddleIntegrator(
#             self.temperature, self.friction_coeff, self.timestep
#         )

#         simulation = Simulation(
#             self.modeller.topology,
#             self.mixed_system,
#             integrator,
#             platformProperties={"Precision": self.openmm_precision},
#         )
#         simulation.context.setPositions(self.modeller.getPositions())
#         # simulation.context.setVelocitiesToTemperature(self.temperature)
#         # logging.info("Minimising energy...")
#         # simulation.context.setParameter("lambda_interpolate", 0)
#         simulation.minimizeEnergy()
#         # minimised_state = simulation.context.getState(
#         #     getPositions=True, getVelocities=True, getForces=True
#         # )
#         # with open(os.path.join(self.output_dir, f"minimised_system.pdb"), "w") as f:
#         #     PDBFile.writeFile(
#         #         self.modeller.topology, minimised_state.getPositions(), file=f
#         #     )

#         reporter = StateDataReporter(
#             file=sys.stdout,
#             reportInterval=interval,
#             step=True,
#             time=True,
#             potentialEnergy=True,
#             temperature=True,
#             speed=True,
#         )
#         simulation.reporters.append(reporter)
#         simulation.reporters.append(
#             PDBReporter(
#                 file=os.path.join(self.output_dir, output_file),
#                 reportInterval=interval,
#                 enforcePeriodicBox=False,
#             )
#         )

#         simulation.step(steps)
