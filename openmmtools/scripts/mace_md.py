import os
import mpiplus
from argparse import ArgumentParser
from openmmtools.openmm_torch.hybrid_md import PureSystem, MixedSystem
from mace import tools
import logging
import torch


logging.getLogger('openmmtools.multistate').setLevel(logging.ERROR)


def main():
    parser = ArgumentParser()

    parser.add_argument("--file", "-f", type=str)
    parser.add_argument(
        "--ml_mol",
        type=str,
        help="either smiles string or file path for the small molecule to be described by MACE",
        default=None,
    )
    parser.add_argument(
        "--run_type", choices=["md", "repex", "neq"], type=str, default="md"
    )
    parser.add_argument("--steps", "-s", type=int, default=10000)
    parser.add_argument("--padding", "-p", default=1.2, type=float)
    parser.add_argument("--nonbondedCutoff", "-c", default=1.0, type=float)
    parser.add_argument("--ionicStrength", "-i", default=0.15, type=float)
    parser.add_argument("--potential", default="mace", type=str)
    parser.add_argument("--temperature", type=float, default=298.15)
    parser.add_argument("--no_minimise", action="store_true")
    parser.add_argument("--pressure", type=float, default=None)
    parser.add_argument(
        "--integrator",
        type=str,
        default="langevin",
        choices=["langevin", "nose-hoover"],
    )
    parser.add_argument(
        "--timestep",
        default=1.0,
        help="integration timestep in femtoseconds",
        type=float,
    )
    parser.add_argument(
        "--extract_nb",
        action="store_true",
        help="If true, extracts non-bonded components of the SM forcefield, adds them to a separate array on the atoms object, writes back out",
    )
    parser.add_argument("--replicas", type=int, default=10)
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        default="output.pdb",
        help="output file for the pdb reporter",
    )
    parser.add_argument("--log_level", default=logging.INFO, type=int)
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument(
        "--output_dir",
        help="directory where all output will be written",
        default="./junk",
    )
    parser.add_argument(
        "--neighbour_list", default="torch_nl", choices=["torch_nl", "torch_nl_n2"]
    )

    # optionally specify box vectors for periodic systems
    parser.add_argument("--box", type=float)

    parser.add_argument("--log_dir", default="./logs")

    parser.add_argument("--restart", action="store_true")
    parser.add_argument(
        "--decouple",
        help="tell the repex constructor to deal with decoupling sterics + electrostatics, instead of lambda_interpolate",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--equil", type=str, choices=["minimise", "gentle"], default="minimise"
    )
    parser.add_argument(
        "--forcefields",
        type=list,
        default=[
            "amber/protein.ff14SB.xml",
            "amber14/DNA.OL15.xml",
        ],
    )
    parser.add_argument("--water_model", type=str, default="tip3p")
    parser.add_argument(
        "--smff",
        help="which version of the openff small molecule forcefield to use",
        default="1.0",
        type=str,
        choices=["1.0", "2.0", "2.0-constrained"],
    )
    parser.add_argument(
        "--interval", help="steps between saved frames", type=int, default=100
    )
    parser.add_argument(
        "--resname",
        "-r",
        help="name of the ligand residue in pdb file",
        default="UNK",
        type=str,
    )
    parser.add_argument("--meta", help="Switch on metadynamics", action="store_true")
    parser.add_argument("--rest2", help="Switch on REST2", action="store_true")
    parser.add_argument(
        "--model_path",
        "-m",
        help="path to the mace model",
        default="tests/test_openmm/MACE_SPICE_larger.model",
    )
    parser.add_argument(
        "--system_type",
        type=str,
        choices=["pure", "hybrid", "decoupled"],
        default="pure",
    )
    parser.add_argument("--mm_only", action="store_true", default=False)
    parser.add_argument("--write_gmx", action="store_true", default=False)
    parser.add_argument(
        "--ml_selection",
        help="specify how the ML subset should be interpreted, either as a resname or a chain ",
        choices=["resname", "chain"],
        default="resname",
    )
    args = parser.parse_args()

    if args.dtype == "float32":
        logging.warning(
            "Running with single precision - this can lead to numerical stability issues"
        )
        torch.set_default_dtype(torch.float32)
        dtype = torch.float32
    elif args.dtype == "float64":
        torch.set_default_dtype(torch.float64)
        dtype = torch.float64
    tools.setup_logger(level=args.log_level, directory=args.log_dir)

    # we don't need to specify the file twice if dealing with just the ligand
    if args.file.endswith(".sdf") and args.ml_mol is None:
        args.ml_mol = args.file
    # ADD THE WATER MODEL TO forcefield list
    if args.water_model == "tip3p":
        args.forcefields.append("amber/tip3p_standard.xml")
    elif args.water_model == "tip4pew":
        args.forcefields.append("amber/tip4pew.xml")
    else:
        raise ValueError(f"Water model {args.water_model} not recognised")

    if args.mm_only and args.system_type == "pure":
        raise ValueError(
            "Cannot run a pure MACE system with only the MM forcefield - please use a hybrid system"
        )
    minimise = False if args.no_minimise else True
    print("Minimise: ", minimise)

    # Only need interpolation when running repex and not decoupling
    interpolate = True if (args.run_type == "repex" and not args.decouple) else False
    print("Interpolate: ", interpolate)

    if args.system_type == "pure":
        # if we're running a pure system, we need to specify the ml_mol, args.file is only useful for metadynamics where we need the topology to extract the right CV atoms
        system = PureSystem(
            file=args.file,
            ml_mol=args.ml_mol,
            model_path=args.model_path,
            potential=args.potential,
            output_dir=args.output_dir,
            temperature=args.temperature,
            pressure=args.pressure,
            dtype=dtype,
            timestep=args.timestep,
            smff=args.smff,
            boxsize=args.box,
            minimise=minimise,
        )

    elif args.system_type == "hybrid":
        system = MixedSystem(
            file=args.file,
            ml_mol=args.ml_mol,
            model_path=args.model_path,
            forcefields=args.forcefields,
            resname=args.resname,
            nnpify_type=args.ml_selection,
            ionicStrength=args.ionicStrength,
            nonbondedCutoff=args.nonbondedCutoff,
            potential=args.potential,
            padding=args.padding,
            temperature=args.temperature,
            dtype=dtype,
            output_dir=args.output_dir,
            smff=args.smff,
            pressure=args.pressure,
            decouple=args.decouple,
            interpolate=interpolate,
            minimise=minimise,
            mm_only=args.mm_only,
            water_model=args.water_model,
            rest2=args.rest2,
            write_gmx=args.write_gmx,
        )
    if args.run_type == "md":
        system.run_mixed_md(
            args.steps,
            args.interval,
            args.output_file,
            run_metadynamics=args.meta,
            integrator_name=args.integrator,
            restart=args.restart,
        )
    elif args.run_type == "repex":
        system.run_repex(
            replicas=args.replicas,
            restart=args.restart,
            steps=args.steps,
            equilibration_protocol=args.equil,
            decouple=args.decouple,
            checkpoint_interval=args.interval
        )
    elif args.run_type == "neq":
        system.run_neq_switching(args.steps, args.interval)
    else:
        raise ValueError(f"run_type {args.run_type} was not recognised")


if __name__ == "__main__":
    main()
