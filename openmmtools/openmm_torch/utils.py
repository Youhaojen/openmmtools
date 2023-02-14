from typing import Optional, List, Iterable
from openff.toolkit.topology import Molecule
from openmm.app import ForceField

from openmmforcefields.generators import SMIRNOFFTemplateGenerator
import openmm


def set_smff(smff: str) -> str:
    """Parse SMFF from command line and initialize the correct open-ff forcefield

    :param str smff: version of the OFF to use
    :raises ValueError: If SMFF version is not recognised
    :return str: the full filename for the OFF xml file
    """
    if smff == "1.0":
        return "openff_unconstrained-1.0.0.offxml"
    elif smff == "2.0":
        return "openff_unconstrained-2.0.0.offxml"
    else:
        raise ValueError(f"Small molecule forcefield {smff} not recognised")


def initialize_mm_forcefield(
    molecule: Optional[Molecule],
    forcefields: List = ["amber/protein.ff14SB.xml"],
    smff: str = "openff_unconstrained-1.0.0.offxml",
) -> ForceField:

    forcefield = ForceField(*forcefields)
    if molecule is not None:
        # Ensure we use unconstrained force field
        smirnoff = SMIRNOFFTemplateGenerator(molecules=molecule, forcefield=smff)
        forcefield.registerTemplateGenerator(smirnoff.generator)
    return forcefield


# taken from peastman/openmm-ml
def remove_bonded_forces(
    system: openmm.System,
    atoms: Iterable[int],
    removeInSet: bool,
    removeConstraints: bool,
) -> openmm.System:
    """Copy a System, removing all bonded interactions between atoms in (or not in) a particular set.

    Parameters
    ----------
    system: System
        the System to copy
    atoms: Iterable[int]
        a set of atom indices
    removeInSet: bool
        if True, any bonded term connecting atoms in the specified set is removed.  If False,
        any term that does *not* connect atoms in the specified set is removed
    removeConstraints: bool
        if True, remove constraints between pairs of atoms in the set

    Returns
    -------
    a newly created System object in which the specified bonded interactions have been removed
    """
    atomSet = set(atoms)

    # Create an XML representation of the System.

    import xml.etree.ElementTree as ET

    xml = openmm.XmlSerializer.serialize(system)
    root = ET.fromstring(xml)

    # This function decides whether a bonded interaction should be removed.

    def shouldRemove(termAtoms):
        return all(a in atomSet for a in termAtoms) == removeInSet

    # Remove bonds, angles, and torsions.

    for bonds in root.findall("./Forces/Force/Bonds"):
        for bond in bonds.findall("Bond"):
            bondAtoms = [int(bond.attrib[p]) for p in ("p1", "p2")]
            if shouldRemove(bondAtoms):
                bonds.remove(bond)
    for angles in root.findall("./Forces/Force/Angles"):
        for angle in angles.findall("Angle"):
            angleAtoms = [int(angle.attrib[p]) for p in ("p1", "p2", "p3")]
            if shouldRemove(angleAtoms):
                angles.remove(angle)
    for torsions in root.findall("./Forces/Force/Torsions"):
        for torsion in torsions.findall("Torsion"):
            torsionAtoms = [int(torsion.attrib[p]) for p in ("p1", "p2", "p3", "p4")]
            if shouldRemove(torsionAtoms):
                torsions.remove(torsion)

    # Optionally remove constraints.

    if removeConstraints:
        for constraints in root.findall("./Constraints"):
            for constraint in constraints.findall("Constraint"):
                constraintAtoms = [int(constraint.attrib[p]) for p in ("p1", "p2")]
                if shouldRemove(constraintAtoms):
                    constraints.remove(constraint)

    # Create a new System from it.

    return openmm.XmlSerializer.deserialize(ET.tostring(root, encoding="unicode"))
