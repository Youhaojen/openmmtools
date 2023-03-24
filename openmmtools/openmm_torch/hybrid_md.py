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
from typing import List, Tuple, Optional
from openmm import (
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
    CustomTorsionForce,
    NoseHooverIntegrator,
)
from openmmtools.integrators import LangevinIntegrator as OpenMMToolsLangevinIntegrator
import matplotlib.pyplot as plt
from openmmtools.integrators import AlchemicalNonequilibriumLangevinIntegrator
from mdtraj.reporters import HDF5Reporter
from mdtraj.geometry.dihedral import indices_phi, indices_psi
from openmm.app import (
    Simulation,
    StateDataReporter,
    PDBReporter,
    DCDReporter,
    CheckpointReporter,
    PDBFile,
    Modeller,
    PME,
)
from openmm.app.metadynamics import Metadynamics, BiasVariable
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

from openmmtools import alchemy


from openmmml.models.macepotential import MACEPotentialImplFactory
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
from abc import ABC, abstractmethod


def get_xyz_from_mol(mol):

    xyz = np.zeros((mol.GetNumAtoms(), 3))
    conf = mol.GetConformer()
    for i in range(conf.GetNumAtoms()):
        position = conf.GetAtomPosition(i)
        xyz[i, 0] = position.x
        xyz[i, 1] = position.y
        xyz[i, 2] = position.z
    return xyz


MLPotential.registerImplFactory("mace", MACEPotentialImplFactory())
MLPotential.registerImplFactory("ani2x", ANIPotentialImplFactory())

logger = logging.getLogger("INFO")


class MACESystemBase(ABC):
    potential: str
    temperature: float
    friction_coeff: float
    timestep: float
    dtype: torch.dtype
    output_dir: str
    openmm_precision: str
    SM_FF: str
    modeller: Modeller
    system: System
    mm_only:bool

    def __init__(
        self,
        file: str,
        model_path: str,
        potential: str,
        output_dir: str,
        temperature: float,
        pressure: Optional[float] = None,
        dtype: torch.dtype = torch.float64,
        friction_coeff: float = 1.0,
        timestep: float = 1.0,
        smff: str = "1.0",
        minimise: bool = True,
        mm_only:bool = False
    ) -> None:
        super().__init__()

        self.file = file
        self.model_path = model_path
        self.potential = potential
        self.temperature = temperature
        self.pressure = pressure
        self.friction_coeff = friction_coeff / picosecond
        self.timestep = timestep * femtosecond
        self.dtype = dtype
        self.output_dir = output_dir
        self.mm_only = mm_only
        self.minimise = minimise
        self.openmm_precision = "Double" if dtype == torch.float64 else "Mixed"
        logger.debug(f"OpenMM will use {self.openmm_precision} precision")

        self.SM_FF = set_smff(smff)
        logger.info(f"Using SMFF {self.SM_FF}")

        os.makedirs(self.output_dir, exist_ok=True)

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
        atoms = read(tmpfile)
        # os.remove(tmpfile)
        return atoms, molecule

    @abstractmethod
    def create_system(self):
        pass

    def run_mixed_md(
        self,
        steps: int,
        interval: int,
        output_file: str,
        restart: bool,
        run_metadynamics: bool = False,
        integrator_name: str = "langevin",
    ):
        """Runs plain MD on the mixed system, writes a pdb trajectory

        :param int steps: number of steps to run the simulation for
        :param int interval: reportInterval attached to reporters
        """
        if integrator_name == "langevin":
            integrator = LangevinMiddleIntegrator(
                self.temperature, self.friction_coeff, self.timestep
            )
        elif integrator_name == "nose-hoover":
            integrator = NoseHooverIntegrator(
                self.temperature, self.friction_coeff, self.timestep
            )
        else:
            raise ValueError(
                f"Unrecognized integrator name {integrator_name}, must be one of ['langevin', 'nose-hoover']"
            )

        if run_metadynamics:
            # if we have initialized from xyz, the topology won't have the information required to identify the cv indices, create from a pdb
            input_file = PDBFile(self.file)
            topology = input_file.getTopology()
            meta = self.run_metadynamics(
                topology=topology
                # cv1_dsl_string=self.cv1_dsl_string, cv2_dsl_string=self.cv2_dsl_string
            )
        # set alchemical state

      

        logger.debug(f"Running mixed MD for {steps} steps")
        simulation = Simulation(
            self.modeller.topology,
            self.system,
            integrator,
            platformProperties={"Precision": self.openmm_precision},
        )
        checkpoint_filepath = os.path.join(self.output_dir, output_file[:-4] + ".chk")
        if restart:
            with open(checkpoint_filepath, "rb") as f:
                logger.info("Loading simulation from checkpoint file...")
                simulation.context.loadCheckpoint(f.read())
        else:

            simulation.context.setPositions(self.modeller.getPositions())
            if self.minimise:
                logging.info("Minimising energy...")
                simulation.minimizeEnergy()
                minimised_state = simulation.context.getState(
                    getPositions=True, getVelocities=True, getForces=True
                )
                with open(os.path.join(self.output_dir, f"minimised_system.pdb"), "w") as f:
                    PDBFile.writeFile(
                        self.modeller.topology, minimised_state.getPositions(), file=f
                    )
            else:
                logger.info("Skipping minimisation step")

        reporter = StateDataReporter(
            file=sys.stdout,
            reportInterval=interval,
            step=True,
            time=True,
            potentialEnergy=True,
            density=True,
            temperature=True,
            speed=True,
        )
        simulation.reporters.append(reporter)
        simulation.reporters.append(
            PDBReporter(
                file=os.path.join(self.output_dir, output_file),
                reportInterval=interval,
                enforcePeriodicBox=False,
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
        # Add an extra hash to any existing checkpoint files
        checkpoint_files = [f for f in os.listdir(self.output_dir) if f.endswith("#")]
        for file in checkpoint_files:
            os.rename(
                os.path.join(self.output_dir, file),
                os.path.join(self.output_dir, f"{file}#"),
            )

        # backup the existing checkpoint file
        if os.path.isfile(checkpoint_filepath):
            os.rename(checkpoint_filepath, checkpoint_filepath + "#")
        checkpoint_reporter = CheckpointReporter(
            file=checkpoint_filepath, reportInterval=interval
        )
        simulation.reporters.append(checkpoint_reporter)

        if run_metadynamics:
            logger.info("Running metadynamics")
            # handles running the simulation with metadynamics
            meta.step(simulation, steps)

            fe = meta.getFreeEnergy()
            fig, ax = plt.subplots(1, 1)
            ax.imshow(fe)
            fig.savefig(os.path.join(self.output_dir, "free_energy.png"))

        else:
            simulation.step(steps)

    def run_repex(
        self,
        replicas: int,
        restart: bool,
        decouple: bool,
        steps: int,
        intervals_per_lambda_window: int = 10,
        steps_per_equilibration_interval: int = 1000,
        equilibration_protocol: str = "minimise",
    ) -> None:
        repex_file_exists = os.path.isfile(os.path.join(self.output_dir, "repex.nc"))
        # even if restart has been set, disable if the checkpoint file was not found, enforce minimising the system
        if not repex_file_exists:
            restart = False
        sampler = RepexConstructor(
            mixed_system=self.system,
            initial_positions=self.modeller.getPositions(),
            intervals_per_lambda_window=2 * replicas,
            steps_per_equilibration_interval=steps_per_equilibration_interval,
            equilibration_protocol=equilibration_protocol,
            temperature=self.temperature * kelvin,
            n_states=replicas,
            restart=restart,
            decouple=decouple,
            mcmc_moves_kwargs={
                "timestep": 1.0 * femtoseconds,
                "collision_rate": 10.0 / picoseconds,
                "n_steps": 1000,
                "reassign_velocities":False,
                "n_restart_attempts": 20,
            },
            replica_exchange_sampler_kwargs={
                "number_of_iterations": steps,
                "online_analysis_interval": 10,
                "online_analysis_minimum_iterations": 10,
            },
            storage_kwargs={
                "storage": os.path.join(self.output_dir, "repex.nc"),
                "checkpoint_interval": 10,
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
            # just run a few steps to make sure the system is in a reasonable conformation

            logging.info(f"Minimised system  in {time.time() - t1} seconds")
            # we want to write out the positions after the minimisation - possibly something weird is going wrong here and it's ending up in a weird conformation

        sampler.run()

    def run_metadynamics(
        # self, topology: Topology, cv1_dsl_string: str, cv2_dsl_string: str
        self,
        topology: Topology,
    ) -> Metadynamics:
        # run well-tempered metadynamics
        mdtraj_topology = mdtraj.Topology.from_openmm(topology)

        cv1_atom_indices = indices_psi(mdtraj_topology)
        cv2_atom_indices = indices_phi(mdtraj_topology)
        # logger.info(f"Selcted cv1 torsion atoms {cv1_atom_indices}")
        # cv2_atom_indices = mdtraj_topology.select(cv2_dsl_string)
        # logger.info(f"Selcted cv2 torsion atoms {cv2_atom_indices}")
        # takes the mixed system parametrised in the init method and performs metadynamics
        # in the canonical case, this should just use the psi-phi backbone angles of the peptide

        cv1 = CustomTorsionForce("theta")
        # cv1.addTorsion(cv1_atom_indices)
        cv1.addTorsion(*cv1_atom_indices[0])
        phi = BiasVariable(cv1, -np.pi, np.pi, biasWidth=0.5, periodic=True)

        cv2 = CustomTorsionForce("theta")
        cv2.addTorsion(*cv2_atom_indices[0])
        psi = BiasVariable(cv2, -np.pi, np.pi, biasWidth=0.5, periodic=True)
        os.makedirs(os.path.join(self.output_dir, "metaD"), exist_ok=True)
        meta = Metadynamics(
            self.system,
            [psi, phi],
            temperature=self.temperature,
            biasFactor=10.0,
            height=1.0 * kilojoule_per_mole,
            frequency=100,
            biasDir=os.path.join(self.output_dir, "metaD"),
            saveFrequency=100,
        )

        return meta

    def decouple_long_range(self, system: System, solute_indices: List) -> System:
        """Create an alchemically modified system with the lambda parameters to decouple the steric and electrostatic components of the forces according to their respective lambda parameters

        :param System system: the openMM system to test
        :param List solute_indices: the list of indices to treat as the alchemical region (i.e. the ligand to be decoupled from solvent)
        :return System: Alchemically modified version of the system with additional lambda parameters for the
        """
        factory = alchemy.AbsoluteAlchemicalFactory(
            alchemical_pme_treatment="exact"
        )

        alchemical_region = alchemy.AlchemicalRegion(
            alchemical_atoms=solute_indices,
            annihilate_electrostatics=True,
            annihilate_sterics=True,
        )
        alchemical_system = factory.create_alchemical_system(system, alchemical_region)

        return alchemical_system

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
            self.system,
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
        protocol_work = integrator.get_protocol_work(dimensionless=True)
        return protocol_work


class MixedSystem(MACESystemBase):
    forcefields: List[str]
    padding: float
    ionicStrength: float
    nonbondedCutoff: float
    resname: str
    nnpify_type: str
    mixed_system: System
    minimise: bool
    water_model: str

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
        output_dir: str,
        decouple: bool,
        interpolate: bool,
        minimise: bool,
        mm_only: bool,
        friction_coeff: float = 1.0,
        timestep: float = 1.0,
        smff: str = "1.0",
        water_model: str = "tip3p",
        pressure: Optional[float] = None,
        cv1: Optional[str] = None,
        cv2: Optional[str] = None,
    ) -> None:
        super().__init__(
            file=file,
            model_path=model_path,
            potential=potential,
            output_dir=output_dir,
            temperature=temperature,
            pressure=pressure,
            dtype=dtype,
            friction_coeff=friction_coeff,
            timestep=timestep,
            smff=smff,
            minimise=minimise,
            mm_only=mm_only,
        )

        self.forcefields = forcefields
        self.padding = padding
        self.ionicStrength = ionicStrength
        self.nonbondedCutoff = nonbondedCutoff
        self.resname = resname
        self.nnpify_type = nnpify_type
        self.cv1 = cv1
        self.cv2 = cv2
        self.water_model = water_model
        self.decouple = decouple
        self.interpolate = interpolate

        logger.debug(f"OpenMM will use {self.openmm_precision} precision")

        # created the hybrid system
        self.create_system(
            file=file,
            ml_mol=ml_mol,
            model_path=model_path,
            pressure=pressure,
        )

    def create_system(
        self,
        file: str,
        model_path: str,
        ml_mol: str,
        pressure: Optional[float],
    ) -> None:
        """Creates the mixed system from a purely mm system

        :param str file: input pdb file
        :param str model_path: path to the mace model
        :return Tuple[System, Modeller]: return mixed system and the modeller for topology + position access by downstream methods
        """
        atoms, molecule = self.initialize_ase_atoms(ml_mol)
        # set the default topology to that of the ml molecule, this will get overwritten below
        # topology = molecule.to_topology().to_openmm()

        # Handle a complex, passed as a pdb file
        if file.endswith(".pdb"):
            input_file = PDBFile(file)
            topology = input_file.getTopology()

            self.modeller = Modeller(input_file.topology, input_file.positions)
            logger.info(
                f"Initialized topology with {len(input_file.positions)} positions"
            )

        # Handle a small molecule/small periodic system, passed as an sdf or xyz
        elif file.endswith(".sdf") or file.endswith(".xyz"):
            input_file = molecule
            topology = molecule.to_topology().to_openmm()
            # Hold positions in nanometers
            positions = get_xyz_from_mol(molecule.to_rdkit()) / 10

            logger.info(f"Initialized topology with {positions.shape} positions")

            self.modeller = Modeller(topology, positions)

        forcefield = initialize_mm_forcefield(
            molecule=molecule, forcefields=self.forcefields, smff=self.SM_FF
        )
        if "tip4p" in self.water_model:
            modeller.addExtraParticles(forcefield)
        self.modeller.addSolvent(
            forcefield,
            model=self.water_model,
            padding=self.padding * nanometers,
            ionicStrength=self.ionicStrength * molar,
            neutralize=True,
        )

        omm_box_vecs = self.modeller.topology.getPeriodicBoxVectors()
        # ensure atoms object has boxvectors taken from the PDB file
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
        
        if not self.decouple:
            if self.mm_only:
                logger.info("Creating MM system")
                self.system = system
            else:
                logger.debug("Creating hybrid system")
                self.system = MixedSystemConstructor(
                    system=system,
                    topology=self.modeller.topology,
                    nnpify_id=self.resname,
                    model_path=model_path,
                    nnp_potential=self.potential,
                    nnpify_type=self.nnpify_type,
                    atoms_obj=atoms,
                    interpolate=self.interpolate,
                    filename=model_path,
                    dtype=self.dtype,
                ).mixed_system

            # optionally, add the alchemical customCVForce for the nonbonded interactions to run ABFE edges
        else:
            if not self.mm_only:
            # TODO: implement decoupled system for VdW/coulomb forces
                logger.info("Creating decoupled system")
                self.system = MixedSystemConstructor(
                    system=system,
                    topology=self.modeller.topology,
                    nnpify_type=self.nnpify_type,
                    nnpify_id=self.resname,
                    nnp_potential=self.potential,
                    model_path=model_path,
                    # cannot have the lambda parameter for this as well as the electrostatics/sterics being decoupled
                    interpolate=False,
                    atoms_obj=atoms,
                    filename=model_path,
                    dtype=self.dtype,
                ).mixed_system

            self.system = self.decouple_long_range(
                self.system,
                solute_indices=get_atoms_from_resname(
                    self.modeller.topology, self.resname, self.nnpify_type
                ),
            )


class PureSystem(MACESystemBase):
    potential: str
    temperature: float
    friction_coeff: float
    timestep: float
    dtype: torch.dtype
    output_dir: str
    openmm_precision: str
    SM_FF: str
    modeller: Modeller
    boxsize: Optional[int]

    def __init__(
        self,
        ml_mol: str,
        model_path: str,
        potential: str,
        output_dir: str,
        temperature: float,
        file: Optional[str] = None,
        boxsize: Optional[int] = None,
        pressure: Optional[float] = None,
        dtype: torch.dtype = torch.float64,
        friction_coeff: float = 1.0,
        timestep: float = 1.0,
        smff: str = "1.0",
        minimise: bool = True,
    ) -> None:

        super().__init__(
            # if file is None, we don't need  to create a topology, so we can pass the ml_mol
            file=ml_mol if file is None else file,
            model_path=model_path,
            potential=potential,
            output_dir=output_dir,
            temperature=temperature,
            pressure=pressure,
            dtype=dtype,
            friction_coeff=friction_coeff,
            timestep=timestep,
            smff=smff,
            minimise=minimise,
        )
        logger.debug(f"OpenMM will use {self.openmm_precision} precision")

        self.boxsize = boxsize

        self.create_system(ml_mol=ml_mol, model_path=model_path, pressure=pressure)

    def create_system(
        self,
        ml_mol: str,
        model_path: str,
        pressure: Optional[float],
    ) -> None:
        """Creates the mixed system from a purely mm system

        :param str file: input pdb file
        :param str model_path: path to the mace model
        :return Tuple[System, Modeller]: return mixed system and the modeller for topology + position access by downstream methods
        """
        # initialize the ase atoms for MACE
        atoms = read(ml_mol)
        if ml_mol.endswith(".xyz"):
            pos = atoms.get_positions() / 10
            box_vectors = atoms.get_cell() / 10
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
                topology.setPeriodicBoxVectors(vectors=box_vectors)

            logger.info(f"Initialized topology with {pos.shape} positions")

            self.modeller = Modeller(topology, pos)

        elif ml_mol.endswith(".sdf"):
            molecule = Molecule.from_file(ml_mol)
            # input_file = molecule
            topology = molecule.to_topology().to_openmm()
            # Hold positions in nanometers
            positions = get_xyz_from_mol(molecule.to_rdkit()) / 10

            # Manually attach periodic box if requested
            if self.boxsize is not None:

                boxvecs = np.eye(3, 3) * self.boxsize
                topology.setPeriodicBoxVectors(boxvecs)

            logger.info(f"Initialized topology with {positions.shape} positions")

            self.modeller = Modeller(topology, positions)
        else:
            raise NotImplementedError

        ml_potential = MLPotential("mace", model_path=model_path)
        self.system = ml_potential.createSystem(
            topology, atoms_obj=atoms, dtype=self.dtype
        )

        if pressure is not None:
            logger.info(
                f"Pressure will be maintained at {pressure} bar with MC barostat"
            )
            barostat = MonteCarloBarostat(pressure * bar, self.temperature * kelvin)
            # barostat.setFrequency(25)  25 timestep is the default
            self.system.addForce(barostat)
