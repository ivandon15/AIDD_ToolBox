"""
A file to convert sdf/pdb/cif to alphafold3 formatted mmcif.

# How to utilize the result:
1. The string returned by other_to_cif() function can be directly used in "userCCD" as below.
{
    "name": "Seq_contains_TBMB",
    "modelSeeds": [
        10,
        42
    ],
    "sequences": [
        {
            "protein": {
                "id": "A",
                "sequence": "XAAAAA",
                "modifications": [
                    {"ptmType": "TBMB", "ptmPosition": 1}
                ],
            }
        }
    ],
    "userCCD": "data_TBMB\n#\n_chem_comp.id TBMB ... _pdbx_chem_comp_descriptor.descriptor 'N[C@@H](C=O)CSCc1cc(CS)cc(CS)c1'\n#",
    "dialect": "alphafold3",
    "version": 1
}

2. If you have multiple custom CCDs, you can modify the file under /path_to_af3_env/lib/python3.11/site-packages/share/libcifpp/components.cif,
append the content in the cif file saved by other_to_cif() function to components.cif. And run build_data again to update the pickle file.
You should pay attention to not provide existed CCD name in components.cif!!

# Error you may encounter
1. "ValueError: First MSA sequence ...": https://github.com/google-deepmind/alphafold3/issues/54
"""
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import ExactMolWt
from alphafold3.data.tools.rdkit_utils import assign_atom_names_from_graph


def load_molecule(input_file):
    """
    Load a molecule from a PDB, SDF, or CIF file.

    Args:
        input_file (str): Path to the input file.

    Returns:
        RDKit Mol object: The loaded molecule.
    """
    file_extension = os.path.splitext(input_file)[-1].lower()

    if file_extension == ".sdf":
        suppl = Chem.SDMolSupplier(input_file, removeHs=False)
        mol = suppl[0]  # Assume only one molecule in the file
    elif file_extension == ".pdb":
        mol = Chem.MolFromPDBFile(input_file, removeHs=False)
    elif file_extension == ".cif":
        mol = Chem.MolFromCIFFile(input_file, removeHs=False)
    else:
        raise ValueError("Unsupported file format. Please provide a PDB, SDF, or CIF file.")

    if mol is None:
        raise ValueError(f"Failed to load molecule from {input_file}.")
    return mol

def process_molecule(mol):
    """
    Standardize and add hydrogens to a molecule.

    Args:
        mol (RDKit Mol object): The input molecule.

    Returns:
        RDKit Mol object: The standardized molecule with hydrogens added.
    """
    mol_with_h = Chem.AddHs(mol)

    # Generate 3D coordinates if not already present
    try: 
        mol_with_h.GetConformer().Is3D()
    except:
        AllChem.EmbedMolecule(mol_with_h, randomSeed=42)

    return mol_with_h


# def smiles2cif(smiles, output_cif_file=None, comp_id="MY-X7F"):
#     mol = Chem.MolFromSmiles(smiles)
#     mol = process_molecule(mol)
#     cif_content = str(mol_to_ccd_cif(mol, component_id=comp_id))
#     if output_cif_file:
#         with open(output_cif_file, "w") as file:
#             file.write(cif_content)

#     return "\\n".join(cif_content.strip().split("\n")),cif_content.strip()

def smiles2cif(smiles, output_cif_file=None, comp_id="MY-X7F"):
    """
    Convert a PDB, SDF, or CIF file to a CIF file with detailed molecular information.

    Args:
        output_cif_file (str): Path to save the output CIF file.
        comp_id (str): Component ID for the CIF file.

    Returns:
        str: Content of the CIF file as a string.
    """
    # Load and process the molecule
    mol = Chem.MolFromSmiles(smiles)
    mol_with_h = process_molecule(mol)
    mol_with_h = assign_atom_names_from_graph(mol_with_h, keep_existing_names=True)

    # Generate molecular information
    smiles = Chem.MolToSmiles(Chem.RemoveHs(mol_with_h), canonical=True)
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol_with_h)
    weight = ExactMolWt(mol_with_h)
    # name = mol.GetProp("_Name") if mol.HasProp("_Name") else "Unknown Ligand"

    # Construct the CIF content
    cif_content = format_cif_header(comp_id, comp_id, formula, weight)
    cif_content += format_cif_atoms(mol_with_h, comp_id)
    cif_content += format_cif_bonds(mol_with_h, comp_id)
    cif_content += format_cif_smiles(smiles)

    # Save CIF content to file
    if output_cif_file:
        with open(output_cif_file, "w") as file:
            file.write(cif_content)

    return "\\n".join(cif_content.strip().split("\n")),cif_content.strip()


# Helper functions for formatting CIF content
def format_cif_header(comp_id, name, formula, weight):
    return f"""
data_{comp_id}
#
_chem_comp.id {comp_id}
_chem_comp.name '{name}'
_chem_comp.type non-polymer
_chem_comp.formula '{formula}'
_chem_comp.mon_nstd_parent_comp_id ?
_chem_comp.pdbx_synonyms ?
_chem_comp.formula_weight {weight:.3f}
#
"""


def format_cif_atoms(mol_with_h, comp_id):
    atom_section = """loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.pdbx_align
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
_chem_comp_atom.pdbx_backbone_atom_flag
_chem_comp_atom.pdbx_n_terminal_atom_flag
_chem_comp_atom.pdbx_c_terminal_atom_flag
_chem_comp_atom.model_Cartn_x
_chem_comp_atom.model_Cartn_y
_chem_comp_atom.model_Cartn_z
_chem_comp_atom.pdbx_model_Cartn_x_ideal
_chem_comp_atom.pdbx_model_Cartn_y_ideal
_chem_comp_atom.pdbx_model_Cartn_z_ideal
_chem_comp_atom.pdbx_component_atom_id
_chem_comp_atom.pdbx_component_comp_id
_chem_comp_atom.pdbx_ordinal
"""
    conf = mol_with_h.GetConformer()
    # backbone_mol = Chem.MolFromSmiles("NCC(=O)-[OD1]")
    # backbone_idxes = mol_with_h.GetSubstructMatches(backbone_mol)[0]
    for i, atom in enumerate(mol_with_h.GetAtoms()):
        atom_id = atom.GetProp('atom_name')
        x, y, z = conf.GetAtomPosition(i)
        aromatic_flag = "Y" if atom.GetIsAromatic() else "N"
        atom_section += f"{comp_id} {atom_id} {atom_id} {atom.GetSymbol().upper()} {atom.GetFormalCharge()} 1 {aromatic_flag} N N N N N {x:.3f} {y:.3f} {z:.3f} {x:.3f} {y:.3f} {z:.3f} {atom_id} {comp_id} {i+1}\n"
    return atom_section


def format_cif_bonds(mol_with_h, comp_id):
    bond_orders = {
        Chem.rdchem.BondType.SINGLE: "SING",
        Chem.rdchem.BondType.DOUBLE: "DOUB",
        Chem.rdchem.BondType.TRIPLE: "TRIP",
        Chem.rdchem.BondType.AROMATIC: "AROM",
    }
    bond_section = """#
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
_chem_comp_bond.pdbx_ordinal
"""
    bond_counter = 1
    for bond in mol_with_h.GetBonds():
        atom1 = bond.GetBeginAtom().GetProp('atom_name')
        atom2 = bond.GetEndAtom().GetProp('atom_name')
        value_order = bond_orders.get(bond.GetBondType(), "SING")
        aromatic_flag = "Y" if bond.GetIsAromatic() else "N"
        bond_section += f"{comp_id} {atom1} {atom2} {value_order} {aromatic_flag} N {bond_counter}\n"
        bond_counter += 1
    return bond_section


def format_cif_smiles(smiles):
    return f"""#
_pdbx_chem_comp_descriptor.type SMILES_CANONICAL
_pdbx_chem_comp_descriptor.descriptor '{smiles}'
#
"""

# cif_output,_ = smiles2cif(smiles, output_cif_file, name)