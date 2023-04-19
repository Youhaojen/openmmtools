import sys
import mdtraj
from ase.io import read
import torch
import time
import numpy as np
from ase import Atoms
from rdkit.Chem.rdmolfiles import MolFromPDBFile, MolFromXYZFile
from openmm.openmm import System
from typing import List, Tuple, Optional
from openmm import (
    LangevinMiddleIntegrator,
    RPMDIntegrator,
    MonteCarloBarostat,
    CustomTorsionForce,
    NoseHooverIntegrator,
)
import re
import matplotlib.pyplot as plt
from openmmtools.integrators import AlchemicalNonequilibriumLangevinIntegrator
from mdtraj.reporters import HDF5Reporter, NetCDFReporter
from mdtraj.geometry.dihedral import indices_phi, indices_psi
from openmm.app import (
    Simulation,
    StateDataReporter,
    PDBReporter,
    DCDReporter,
    CheckpointReporter,
    PDBFile,
    Modeller,
    CutoffNonPeriodic,
    PME,
    HBonds,
)
from openmm.app.metadynamics import Metadynamics, BiasVariable
from openmm.app.topology import Topology
from openmm.app.element import Element
from openmm.unit import (
    kelvin,
    picosecond,
    kilocalorie_per_mole,
    femtosecond,
    kilojoule_per_mole,
    picoseconds,
    femtoseconds,
    bar,
    nanometers,
    molar,
    angstrom,
)
from openff.toolkit.topology import Molecule
from openff.toolkit import ForceField

from openmmtools import alchemy
from atmmetaforce import *

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
    ForceReporter,
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
    mm_only: bool

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
        mm_only: bool = False,
        rest2: bool = False,
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
        self.rest2 = rest2
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
        elif integrator_name == "rpmd":
            # note this requires a few changes to how we set positions
            integrator = RPMDIntegrator(
                5, self.temperature, self.friction_coeff, self.timestep
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
        if restart and os.path.isfile(checkpoint_filepath):
            with open(checkpoint_filepath, "rb") as f:
                logger.info("Loading simulation from checkpoint file...")
                simulation.context.loadCheckpoint(f.read())
        else:
            if isinstance(integrator, RPMDIntegrator):
                for copy in range(integrator.getNumCopies()):
                    integrator.setPositions(copy, self.modeller.getPositions())
            else:
                simulation.context.setPositions(self.modeller.getPositions())
                # rpmd requires that the integrator be used to set positions
            if self.minimise:
                logging.info("Minimising energy...")
                simulation.minimizeEnergy(maxIterations=10)
                if isinstance(integrator, RPMDIntegrator):
                    minimised_state = integrator.getState(
                        0, getPositions=True, getVelocities=True, getForces=True
                    )
                else:
                    minimised_state = simulation.context.getState(
                        getPositions=True, getVelocities=True, getForces=True
                    )

                with open(
                    os.path.join(self.output_dir, "minimised_system.pdb"), "w"
                ) as f:
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
            totalEnergy=True,
            potentialEnergy=True,
            density=True,
            volume=True,
            temperature=True,
            speed=True,
            progress=True,
            totalSteps=steps,
        )
        simulation.reporters.append(reporter)
        # keep periodic box off to make quick visualisation easier
        simulation.reporters.append(
            PDBReporter(
                file=os.path.join(self.output_dir, output_file),
                reportInterval=interval,
                enforcePeriodicBox=False,
            )
        )
        # add force reporter
        simulation.reporters.append(
            ForceReporter(
                file=os.path.join(self.output_dir, "forces.txt"),
                reportInterval=interval,
            )
        )
        # we need this to hold the box vectors for NPT simulations
        netcdf_reporter = NetCDFReporter(
            file=os.path.join(self.output_dir, output_file[:-4] + ".nc"),
            reportInterval=interval,
        )
        simulation.reporters.append(netcdf_reporter)
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
        checkpoint_interval: int = 10,
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
                "checkpoint_interval": checkpoint_interval,
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
        factory = alchemy.AbsoluteAlchemicalFactory(alchemical_pme_treatment="exact")

        alchemical_region = alchemy.AlchemicalRegion(
            alchemical_atoms=solute_indices,
            annihilate_electrostatics=True,
            annihilate_sterics=True,
        )
        alchemical_system = factory.create_alchemical_system(system, alchemical_region)

        return alchemical_system

    def run_atm(self, steps: int, interval: int) -> None:

        lmbd = 0.5
        lambda1 = lmbd
        lambda2 = lmbd
        alpha = 0.0 / kilocalorie_per_mole
        u0 = 0.0 * kilocalorie_per_mole
        w0coeff = 0.0 * kilocalorie_per_mole
        umsc = 100.0 * kilocalorie_per_mole
        ubcore = 50.0 * kilocalorie_per_mole
        acore = 0.062500
        direction = 1.0

        rcpt_resid = 1
        lig1_resid = 2
        lig2_resid = 3
        displ = [22.0, 22.0, 22.0]

        [displ[i] for i in range(3)] * angstrom
        lig1_restr_offset = [0.0 for i in range(3)] * angstrom
        lig2_restr_offset = [displ[i] for i in range(3)] * angstrom
        # TODO: how to select ligand ref atoms
        refatoms_lig1 = [8, 6, 4]
        refatoms_lig2 = [3, 5, 1]

        atm_utils = ATMMetaForceUtils(self.system)

        self.modeller.topology.getNumAtoms()

        rcpt_atoms = []
        for at in self.modeller.topology.atoms():
            if int(at.residue.id) == rcpt_resid:
                rcpt_atoms.append(at.index)

        lig1_atoms = []
        for at in self.modeller.topology.atoms():
            if int(at.residue.id) == lig1_resid:
                lig1_atoms.append(at.index)

        lig2_atoms = []
        for at in self.modeller.topology.atoms():
            if int(at.residue.id) == lig2_resid:
                lig2_atoms.append(at.index)

        rcpt_atom_restr = rcpt_atoms
        lig1_atom_restr = lig1_atoms
        lig2_atom_restr = lig2_atoms

        kf = (
            25.0 * kilocalorie_per_mole / angstrom**2
        )  # force constant for Vsite CM-CM restraint
        r0 = 5 * angstrom  # radius of Vsite sphere

        # these can be 'None" if not using orientational restraints

        # Vsite restraint for lig1
        atm_utils.addVsiteRestraintForceCMCM(
            lig_cm_particles=lig1_atom_restr,
            rcpt_cm_particles=rcpt_atom_restr,
            kfcm=kf,
            tolcm=r0,
            offset=lig1_restr_offset,
        )

        # Vsite restraint for lig2 (offset into the bulk position)
        atm_utils.addVsiteRestraintForceCMCM(
            lig_cm_particles=lig2_atom_restr,
            rcpt_cm_particles=rcpt_atom_restr,
            kfcm=kf,
            tolcm=r0,
            offset=lig2_restr_offset,
        )

        # alignment restraints between lig1 and lig2
        lig1_ref_atoms = [refatoms_lig1[i] + lig1_atoms[0] for i in range(3)]
        lig2_ref_atoms = [refatoms_lig2[i] + lig2_atoms[0] for i in range(3)]
        atm_utils.addAlignmentForce(
            liga_ref_particles=lig1_ref_atoms,
            ligb_ref_particles=lig2_ref_atoms,
            kfdispl=2.5 * kilocalorie_per_mole / angstrom**2,
            ktheta=10.0 * kilocalorie_per_mole,
            kpsi=10.0 * kilocalorie_per_mole,
            offset=lig2_restr_offset,
        )

        # receptor positional restraints, C-atoms of lower cup of the TEMOA host
        fc = 25.0 * kilocalorie_per_mole / angstrom**2
        tol = 0.5 * angstrom
        carbon = re.compile("^C.*")
        posrestr_atoms = []
        for at in self.modeller.topology.atoms():
            if (
                int(at.residue.id) == rcpt_resid
                and carbon.match(at.name)
                and at.index < 40
            ):
                posrestr_atoms.append(at.index)
        atm_utils.addPosRestraints(posrestr_atoms, self.modeller.positions, fc, tol)

        # create ATM Force
        atmforcegroup = 2
        nonbonded_force_group = 1
        atm_utils.setNonbondedForceGroup(nonbonded_force_group)
        atmvariableforcegroups = [nonbonded_force_group]
        atmforce = ATMMetaForce(
            lambda1,
            lambda2,
            alpha * kilojoules_per_mole,
            u0 / kilojoules_per_mole,
            w0coeff / kilojoules_per_mole,
            umsc / kilojoules_per_mole,
            ubcore / kilojoules_per_mole,
            acore,
            direction,
            atmvariableforcegroups,
        )
        for at in self.modeller.topology.atoms():
            atmforce.addParticle(at.index, 0.0, 0.0, 0.0)
        for i in lig1_atoms:
            atmforce.setParticleParameters(
                i, i, displ[0] * angstrom, displ[1] * angstrom, displ[2] * angstrom
            )
        for i in lig2_atoms:
            atmforce.setParticleParameters(
                i, i, -displ[0] * angstrom, -displ[1] * angstrom, -displ[2] * angstrom
            )
        atmforce.setForceGroup(atmforcegroup)
        self.system.addForce(atmforce)
        print("Using ATM Meta Force plugin version = %s" % ATMMETAFORCE_VERSION)

        # setup integrator
        temperature = 300 * kelvin
        frictionCoeff = 0.5 / picosecond
        MDstepsize = 0.001 * picosecond

        # add barostat but turned off, needed to load checkopoint file written with NPT
        barostat = MonteCarloBarostat(1 * bar, temperature)
        barostat.setFrequency(0)  # disabled
        self.system.addForce(barostat)

        integrator = LangevinMiddleIntegrator(
            temperature / kelvin,
            frictionCoeff / (1 / picosecond),
            MDstepsize / picosecond,
        )
        integrator.setIntegrationForceGroups({0, atmforcegroup})

        # platform_name = 'OpenCL'
        # platform_name = 'Reference'
        # platform_name = 'CUDA'
        # platform = Platform.getPlatformByName("CUDA")

        # properties = {}
        # properties["Precision"] = "mixed"

        simulation = Simulation(self.modeller.topology, self.system, integrator)
        print("Using platform %s" % simulation.context.getPlatform().getName())
        simulation.context.setPositions(self.modeller.positions)

        # one preliminary energy evaluation seems to be required to init the energy routines
        state = simulation.context.getState(getEnergy=True, groups={0, atmforcegroup})
        state.getPotentialEnergy()

        # we should do this

        # override ATM parameters from checkpoint file
        simulation.context.setParameter(atmforce.Lambda1(), lambda1)
        simulation.context.setParameter(atmforce.Lambda2(), lambda2)
        simulation.context.setParameter(atmforce.Alpha(), alpha * kilojoules_per_mole)
        simulation.context.setParameter(atmforce.U0(), u0 / kilojoules_per_mole)
        simulation.context.setParameter(atmforce.W0(), w0coeff / kilojoules_per_mole)
        simulation.context.setParameter(atmforce.Umax(), umsc / kilojoules_per_mole)
        simulation.context.setParameter(atmforce.Ubcore(), ubcore / kilojoules_per_mole)
        simulation.context.setParameter(atmforce.Acore(), acore)
        simulation.context.setParameter(atmforce.Direction(), direction)

        state = simulation.context.getState(getEnergy=True, groups={0, atmforcegroup})
        print("Potential Energy = ", state.getPotentialEnergy())

        print("Leg1 production at lambda = %f ..." % lmbd)

        stepId = 5000
        totalSteps = 50000
        loopStep = int(totalSteps / stepId)
        simulation.reporters.append(
            StateDataReporter(
                sys.stdout, stepId, step=True, potentialEnergy=True, temperature=True
            )
        )
        simulation.reporters.append(
            DCDReporter(os.path.join(self.output_dir, "output" + ".dcd"), stepId)
        )

        binding_file = "energies" + ".out"
        f = open(os.path.join(self.output_dir, binding_file), "w")

        for i in range(loopStep):
            simulation.step(stepId)
            state = simulation.context.getState(
                getEnergy=True, groups={0, atmforcegroup}
            )
            pot_energy = (state.getPotentialEnergy()).value_in_unit(
                kilocalorie_per_mole
            )
            pert_energy = (
                atmforce.getPerturbationEnergy(simulation.context)
            ).value_in_unit(kilocalorie_per_mole)
            l1 = simulation.context.getParameter(atmforce.Lambda1())
            l2 = simulation.context.getParameter(atmforce.Lambda2())
            a = simulation.context.getParameter(atmforce.Alpha()) / kilojoules_per_mole
            umid = simulation.context.getParameter(atmforce.U0()) * kilojoules_per_mole
            w0 = simulation.context.getParameter(atmforce.W0()) * kilojoules_per_mole
            print(
                "%f %f %f %f %f %f %f %f %f"
                % (
                    temperature / kelvin,
                    lmbd,
                    l1,
                    l2,
                    a * kilocalorie_per_mole,
                    umid / kilocalorie_per_mole,
                    w0 / kilocalorie_per_mole,
                    pot_energy,
                    pert_energy,
                ),
                file=f,
            )
            f.flush()

        print("SaveState ...")
        simulation.saveState("final_state" + "-out.xml")

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
        rest2: bool,
        friction_coeff: float = 1.0,
        timestep: float = 1.0,
        smff: str = "1.0",
        water_model: str = "tip3p",
        pressure: Optional[float] = None,
        cv1: Optional[str] = None,
        cv2: Optional[str] = None,
        write_gmx: bool = False,
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
            rest2=rest2,
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
        self.write_gmx = write_gmx

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
        if ml_mol is not None:
            atoms, molecule = self.initialize_ase_atoms(ml_mol)
        else:
            atoms, molecule = None, None
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
        # this should also handle generating smirnoff parameters for something like an octa-acid, where this is still to be handled by the MM forcefield, but needs parameters generated
        elif file.endswith(".sdf") or file.endswith(".xyz"):

            # handle the case where the receptor and ligand are both passed as different sdf files:
            if ml_mol != file:
                logger.info("Combining and parametrising 2 sdf files...")
                # load the receptor
                receptor_as_molecule = Molecule.from_file(file)

                # create modeller from this
                self.modeller = Modeller(
                    receptor_as_molecule.to_topology().to_openmm(),
                    get_xyz_from_mol(receptor_as_molecule.to_rdkit()) / 10,
                )
                # combine with modeller for the ml_mol
                ml_mol_modeller = Modeller(
                    molecule.to_topology().to_openmm(),
                    get_xyz_from_mol(molecule.to_rdkit()) / 10,
                )

                self.modeller.add(ml_mol_modeller.topology, ml_mol_modeller.positions)
                # send both to the forcefield initializer
                molecule = [molecule, receptor_as_molecule]

            else:

                input_file = molecule
                topology = molecule.to_topology().to_openmm()
                # Hold positions in nanometers
                positions = get_xyz_from_mol(molecule.to_rdkit()) / 10

                logger.info(f"Initialized topology with {positions.shape} positions")

                self.modeller = Modeller(topology, positions)

        forcefield = initialize_mm_forcefield(
            molecule=molecule, forcefields=self.forcefields, smff=self.SM_FF
        )
        if self.write_gmx:
            from openff.interchange import Interchange

            interchange = Interchange.from_smirnoff(
                topology=molecule.to_topology(), force_field=ForceField(self.SM_FF)
            )
            interchange.to_top(os.path.join(self.output_dir, "topol.top"))
            interchange.to_gro(os.path.join(self.output_dir, "conf.gro"))
        if self.padding > 0:
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
            if atoms is not None:
                atoms.set_cell(
                    [
                        omm_box_vecs[0][0].value_in_unit(angstrom),
                        omm_box_vecs[1][1].value_in_unit(angstrom),
                        omm_box_vecs[2][2].value_in_unit(angstrom),
                    ]
                )
        # else:
        # this should be a large enough box
        # run a non-periodic simulation
        # self.modeller.topology.setPeriodicBoxVectors([[5, 0, 0], [0, 5, 0], [0, 0, 5]])

        system = forcefield.createSystem(
            self.modeller.topology,
            nonbondedMethod=PME
            if self.modeller.topology.getPeriodicBoxVectors() is not None
            else CutoffNonPeriodic,
            nonbondedCutoff=self.nonbondedCutoff * nanometer,
            constraints=None if "unconstrained" in self.SM_FF else HBonds,
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

        # if self.write_gmx:
        #     # write the openmm system to gromacs top/gro with parmed
        #     from parmed.openmm import load_topology

        #     parmed_structure = load_topology(self.modeller.topology, system)
        #     parmed_structure.save(os.path.join(self.output_dir, "topol_full.top"), overwrite=True)
        #     parmed_structure.save(os.path.join(self.output_dir, "conf_full.gro"), overwrite=True)
        #     raise KeyboardInterrupt

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
                    T_high=450 * kelvin if self.rest2 else 300 * kelvin,
                    T_low=300 * kelvin,
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
