import os, json, copy, pathlib
import pandas as pd
from tqdm import tqdm
from pymol import cmd
from Bio.SeqUtils import seq1
from rdkit import Chem
from rdkit.Chem import AllChem
from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.data.tools.rdkit_utils import mol_to_ccd_cif

# Load chemical components dictionary
ccd = chemical_components.cached_ccd()
# Build the smi2ccd mapping
smi2ccd = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"smi2ccd.json"),"r"))

def canonical_smiles(smiles):
    """Convert a SMILES string to its canonical form."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def find_ccd(smiles):
    smiles = canonical_smiles(smiles)
    return smi2ccd.get(smiles, None)


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


def smiles2cif(smiles, output_cif_file=None, comp_id="MY-X7F"):
    """
    Utilizing af3 rdkit tools to translate smiles to af3-readable cif,
    better using output_cif_file if you need to know the name of atom
    """
    mol = process_molecule(Chem.MolFromSmiles(smiles))
    cif_content = str(mol_to_ccd_cif(mol, component_id=comp_id))
    if output_cif_file:
        with open(output_cif_file, "w") as file:
            file.write(cif_content)

    return "\\n".join(cif_content.strip().split("\n")),cif_content.strip()


def _format_bondedAtomPairs(cyclic_pos, linker_pos, pep_id, linker_id):
    """
    Generate a list of bonded atom pairs from cyclic and linker positions.
    
    Args:
        cyclic_pos (str): Cyclic position data in the format '1|SG,10|SG;2|C1,3|C1...'.
        linker_pos (str): Linker position data, corresponding to cyclic_pos if exists.
        pep_id (str): Identifier for the peptide chain.
        linker_id (str): Identifier for the linker chain, if applicable.
    
    Returns:
        list: A list of bonded atom pairs, formatted as [[chain, position, atom], [chain, position, atom]].
    """
    bonded_pairs = []
    cyclic_groups = [group.split(",") for group in cyclic_pos.split(";")]
    linker_groups = [group.split(",") for group in linker_pos.split(";")] if linker_id else []

    for i, pep_atoms in enumerate(cyclic_groups):
        pep_atoms = [atom.split("|") for atom in pep_atoms]

        if not linker_id or not linker_groups[i]:
            bonded_pairs.append([[pep_id, int(pep_atoms[0][0]), pep_atoms[0][1]],
                                 [pep_id, int(pep_atoms[1][0]), pep_atoms[1][1]]])
        else:
            link_atoms = [atom.split("|") for atom in linker_groups[i]]
            for pep_atom, link_atom in zip(pep_atoms, link_atoms):
                bonded_pairs.append([[pep_id, int(pep_atom[0]), pep_atom[1]],
                                     [linker_id, int(link_atom[0]), link_atom[1]]])
    
    return bonded_pairs


def _lst2seq(lst, lookup_dict=None):
    """
    Convert a list of amino acid triplets into a sequence string, while handling custom components.
    
    Args:
        lst (list): List of amino acid triplets.
        lookup_dict (dict): Dictionary mapping custom components to their SMILES representations.
    
    Returns:
        tuple: (sequence string, list of user-defined CCDs, dictionary of position mappings)
    """
    seq_aa = []
    position_dict = {}
    userCCDs = set()
    
    for id, aa3 in enumerate(lst, start=1):
        aa1 = seq1(aa3)
        if aa1 == "X":
            ccd_info = ccd.get(aa3)
            if not ccd_info:
                if lookup_dict is None:
                    raise ValueError("Contains new UAA, you must provide lookup table.")
                if aa3 not in lookup_dict.keys():
                    raise ValueError(f"Lookup table missing new component {aa3} smiles")
                # try to use smiles find existed ccdCode
                smiles = lookup_dict[aa3]
                ccdCode = find_ccd(smiles)
                if ccdCode:
                    ccd_info = ccd.get(ccdCode)
                    aa1 = ccd_info['_chem_comp.one_letter_code'][0]
                    aa1 = "X" if aa1 == "?" else aa1
                else:
                    # need custom ccdCode
                    _, userCCD_str = smiles2cif(smiles, comp_id=aa3)
                    userCCDs.add(userCCD_str)
            else:
                aa1 = ccd_info['_chem_comp.one_letter_code'][0]
                aa1 = "X" if aa1 == "?" else aa1
            position_dict[id] = aa3
        seq_aa.append(aa1)
    
    return "".join(seq_aa), userCCDs, position_dict


def df2dict(df, lookuptable=None):
    """
    Convert a DataFrame into a structured dictionary representation for biomolecular sequences.

    This function processes a given DataFrame containing biomolecular sequence information 
    and converts it into a nested dictionary format. It supports handling standard protein sequences, 
    peptides containing unnatural amino acids (UAAs), and optional linkers.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing sequence data. The DataFrame must have 
            columns starting with "chain" for normal protein sequences, and optionally columns 
            starting with "P" for peptides with UAAs.
        lookuptable (str, optional): A CSV file path containing a lookup table for converting CCD 
            codes to SMILES representations. If provided, it should have "CCD" and "smiles" columns.

    Returns:
        dict: A structured dictionary containing sequence data. The dictionary structure includes:
            - "chains": A dictionary of protein chains and their sequences.
            - "linker" (optional): Linker information if applicable.
            - "cyclic_pos" and "linker_pos" (optional): Position information if a linker is present.
            - "bondedAtomPairs" (optional): Bonding information if a linker exists.
            - "userCCD" (optional): Custom CCD information for UAAs or linkers.

    Raises:
        ValueError: If required columns ("chain*" or necessary linker-related columns) are missing.
        Exception: If there is an issue processing the lookup table or row entries.
    """
    info_dict = {}
    lookup_dict = None
    chain_cols = [col for col in df.columns if col.startswith("chain")]
    if chain_cols == []:
        raise ValueError("Your candidate sheet columns must meet the requirements!")
    
    # usually used when peptide contains UAA
    p_cols = [col for col in df.columns if col.startswith("P")]
    linker_cols = [col for col in df.columns if col.startswith("linker")]
    cyclic_pos_cols = [col for col in df.columns if col.startswith("pos_cyclic")]
    linker_pos_cols = [col for col in df.columns if col.startswith("pos_linker")]
    if lookuptable:
        lookup_df = pd.read_csv(lookuptable)
        try:
            lookup_dict = dict(zip(lookup_df['CCD'], lookup_df['smiles']))
        except Exception as e:
            print("Your lookuptable sheet columns must meet the requirements!")

    for _, row in df.iterrows():
        # try:
        entry = {}
        # only normal sequences
        chains = {
            col.split("chain")[-1]: {
                "type": "protein",
                "sequence": row[col], 
                "modifications":[]
            } for col in chain_cols if not pd.isna(row[col])
        }
        entry["bondedAtomPairs"] = []

        # UAA contains           
        if p_cols:
            
            # deal with linear peptide part
            pepseq, userCCDs, pTM_dict = _lst2seq(
                [row[col] for col in p_cols if not pd.isna(row[col])],
                lookup_dict
            )
            if pepseq:
                pep_id = chr(ord(list(chains.keys())[-1])+1)
                chains[pep_id] = {
                    "type": "protein", 
                    "sequence": pepseq, 
                    "modifications":[
                        {
                            "ptmType": aa3, 
                            "ptmPosition": idx
                        } for idx, aa3 in pTM_dict.items()
                    ]
                }

                # deal with linker part, but link may not exist, or may have multiple
                for linker_col, cyclic_pos_col, linker_pos_col in zip(
                        linker_cols, 
                        cyclic_pos_cols,
                        linker_pos_cols
                    ):
                    entry_linker = row.get(linker_col, "")
                    entry_cyclic_pos = row.get(cyclic_pos_col, "")
                    entry_linker_pos = row.get(linker_pos_col, "")
                    if not pd.isna(entry_linker) and entry_linker:
                        _, linkeruserCCD = smiles2cif(
                            lookup_dict[entry_linker], 
                            comp_id=entry_linker
                        )
                        userCCDs.add(linkeruserCCD)
                        linker_id = chr(ord(list(chains.keys())[-1])+1)
                        chains[linker_id] = {"type": "ligand", "ccdCodes": entry_linker}

                        # if linker exists, then the following two must exists
                        if pd.isna(entry_cyclic_pos) or pd.isna(entry_linker_pos):
                            raise ValueError(
                                f"ID {row['ID']}: If 'linker' exists, 'cyclic_pos' and "
                                "'linker_pos' must exist."
                            )
                        
                        # and we need record bondedAtomPairs
                        entry["bondedAtomPairs"].extend(
                            _format_bondedAtomPairs(entry_cyclic_pos, entry_linker_pos, pep_id, linker_id)
                        )
                if userCCDs:
                    entry['userCCD'] = "\n".join(list(userCCDs))

        entry['chains'] = chains
        info_dict[row["ID"]] = entry
    
        # except ValueError as e:
        #     with open("error_log.txt", "a") as log_file:
        #         log_file.write(f"Error processing row: {row['ID']}. Error: {e}\n")
        #         print("Error log was outputed at error_log.txt")
        # except Exception as e:
        #     with open("error_log.txt", "a") as log_file:
        #         log_file.write(f"Unexpected error processing row: {row['ID']}. Error: {e}\n")
        #         print("Error log was outputed at error_log.txt")
    return info_dict


def dict2afjsons(
        info_dict, 
        template_file=None, 
        output_folder="af_jsons", 
        modelSeeds=[10,42], 
        mut_peptide=False
    ):
    """
    Convert a structured sequence dictionary into AlphaFold-compatible JSON files.

    This function takes the dictionary produced by `df2dict` and converts each entry into 
    a JSON file formatted for AlphaFold3. It supports the use of template files and 
    sequence modifications.

    Args:
        info_dict (dict): A dictionary containing sequence data in the format produced by `df2dict`.
        template_file (str or list, optional): A template JSON file or a list of template files 
            for initializing output JSONs. If not provided, default structures are created.
        output_folder (str, optional): The directory where the generated JSON files will be saved. 
            Defaults to "af_jsons".
        modelSeeds (list, optional): A list of model seeds for AlphaFold. Defaults to [10, 42].
        mut_peptide (bool, optional): If True, allows peptide mutations where sequences have the 
            same length as the template but different residues.

    Returns:
        None: The function writes JSON files to the specified `output_folder`.

    Raises:
        ValueError: If protein sequences in the template do not match those in `info_dict`, unless 
            `mut_peptide` is enabled for same-length modifications.
    """
    os.makedirs(output_folder, exist_ok=True)

    if template_file and isinstance(template_file, str):
        template_file = [template_file]*len(info_dict)

    for _, (entry_id, entry_data) in tqdm(enumerate(info_dict.items())):
        
        # keep entry_id is a string
        entry_id = str(entry_id)
        if template_file:
            with open(template_file[_], "r") as f:
                entity = json.load(f)
            entity["name"] = entry_id
        else:
            entity = {
                "name": entry_id,
                "sequences": [],
                "modelSeeds": modelSeeds,
                "dialect": "alphafold3",
                "version": 1
            }

        # deal with protein chains
        updated_sequences = []
        existing_sequences = {
            seq["protein"]["id"]: seq 
            for seq in entity.get("sequences", []) 
            if "protein" in seq
        }

        for chain_id, chain_data in entry_data["chains"].items():
            if chain_data["type"] == "protein":
                
                # if template exists, then seq_data will contain msa
                seq_data = existing_sequences.get(chain_id,{"protein":{"id": chain_id}})
                original_seq = seq_data["protein"].get("sequence", "")
                new_seq = chain_data["sequence"]
                if original_seq and original_seq != new_seq:
                    if len(original_seq) > 50:
                        raise ValueError(
                            f"Protein {chain_id} in template did not match "
                            "the one in candidates."
                        )
                    elif mut_peptide and len(original_seq) == len(new_seq):
                        # same length peptide
                        seq_data["protein"]["unpairedMsa"] = f">query\n{chain_data['sequence']}\n"
                        seq_data["protein"]["pairedMsa"] = f">query\n{chain_data['sequence']}\n"
                    else:
                        raise ValueError(
                            f"entry_id: {entry_id} peptide {chain_id} in template "
                            "did not match the one in candidates, not even mutation peptides."
                        )
                    
                # Update sequence and modifications
                seq_data["protein"]["sequence"] = new_seq
                if "modifications" in chain_data:
                    seq_data["protein"]["modifications"] = chain_data["modifications"]

                updated_sequences.append(seq_data)
            else:  
                # new linker
                updated_sequences.append({
                    "ligand": {
                        "id": chain_id,
                        "ccdCodes": [chain_data["ccdCodes"]]
                    }
                })

        entity["sequences"] = updated_sequences
        if entry_data.get("bondedAtomPairs"):
            entity["bondedAtomPairs"] = entry_data.get("bondedAtomPairs")
        if entry_data.get("userCCD"):
            entity["userCCD"] = entry_data.get("userCCD")
        file_path = os.path.join(output_folder, f"{entry_id}.json")
        with open(file_path, "w") as f:
            json.dump(entity, f, indent=4)

    print(f"Updated {len(info_dict)} JSON files in {output_folder}")


def dataframe2afjsons(
        df, 
        lookuptable=None, 
        template_file=None, 
        output_folder="af_jsons", 
        modelSeeds=[10,42], 
        mut_peptide=False
    ):
    """
    lookuptable: csv path
    """
    info_dict = df2dict(df, lookuptable)
    dict2afjsons(info_dict, template_file, output_folder,  modelSeeds, mut_peptide)


def addpep2afjson(template_file, output_folder="output_jsons", pepnum=5):
    """
    Duplicate peptide chains in an AlphaFold JSON template to create a complex with multiple 
    peptides.

    Args:
        template_file (str): Path to the input JSON template file containing AlphaFold sequences.
        output_folder (str, optional): Directory where the modified JSON file will be saved.
                                       Defaults to "output_jsons".
        pepnum (int, optional): Number of peptide chains to generate (including the original one).
                                Must be ≥ 2. Defaults to 5.

    Returns:
        None: The function writes the updated JSON file to the specified `output_folder`.

    Raises:
        ValueError: If `pepnum` is less than 2, as the function is unnecessary for a single 
                    peptide.

    Notes:
        - If the template contains ligands, each duplicated peptide will be assigned a new ligand.
        - The `bondedAtomPairs` field is updated accordingly to reflect new chain IDs.
        - Chain IDs are assigned sequentially using ASCII characters.
    """

    os.makedirs(output_folder, exist_ok=True)

    with open(template_file, "r") as f:
        entity = json.load(f)

    updated_sequences = entity['sequences']
    if pepnum < 2:
        print("If you only need to predict complex with 1 peptide, no need using this function.")
        return

    bondedAtomPairs = copy.deepcopy(entity["bondedAtomPairs"])

    last_protein = None
    ligands = []
    for item in updated_sequences:
        if "protein" in item:
            last_protein = item
        elif "ligand" in item:
            ligands.append(item)
    if ligands:
        for i in range(1, pepnum):
            id_mapping = {}
            new_protein = copy.deepcopy(last_protein)
            new_protein["protein"]["id"] = f"{chr(ord(last_protein['protein']['id']) + (len(ligands)+1)*i)}"
            updated_sequences.append(new_protein)
            id_mapping[last_protein['protein']['id']] = new_protein["protein"]["id"]
            for ligand in ligands:
                new_ligand = copy.deepcopy(ligand)
                new_ligand["ligand"]["id"] = f"{chr(ord(ligand['ligand']['id']) + (len(ligands)+1)*i)}"
                updated_sequences.append(new_ligand)
                id_mapping[ligand['ligand']['id']] = new_ligand["ligand"]["id"]
            for bond in entity["bondedAtomPairs"]:
                cp_bond = copy.deepcopy(bond)
                cp_bond[0][0] = id_mapping[cp_bond[0][0]]
                cp_bond[1][0] = id_mapping[cp_bond[1][0]]
                bondedAtomPairs.append(cp_bond)

    else:
        # if only peptide without linker
        for i in range(1, pepnum):
            new_protein = copy.deepcopy(last_protein)
            new_protein["protein"]["id"] = f"{chr(ord(last_protein['protein']['id']) + i)}"
            updated_sequences.append(new_protein)
            for bond in entity["bondedAtomPairs"]:
                cp_bond = copy.deepcopy(bond)
                if cp_bond[0][0] == last_protein["protein"]["id"]:
                    cp_bond[0][0] = new_protein["protein"]["id"]
                    cp_bond[1][0] = new_protein["protein"]["id"]
                bondedAtomPairs.append(cp_bond)
    entity["bondedAtomPairs"] = bondedAtomPairs
    entity["sequences"] = updated_sequences
    if entity.get("userCCD", None):
        entity["userCCD"] = entity["userCCD"]
    file_path = os.path.join(output_folder, f"{entity['name']}.json")
    with open(file_path, "w") as f:
        json.dump(entity, f, indent=4)


def cifs2afjsons(cif_folder, output_folder="af_jsons"):
    """
    Converting cif files in one folder to alphafold input json type.
    Basically used in cif file which contain UAA (PTM).
    """
    count = 0
    for mmcif_name in tqdm(os.listdir(cif_folder)):
        mmcif_file_path = os.path.join(cif_folder, mmcif_name)
        try:
            with open(mmcif_file_path) as f:
                mmcif = f.read()

            alphafold_input = folding_input.Input.from_mmcif(
                mmcif, ccd=chemical_components.cached_ccd()
            )

            with open(os.path.join(
                        output_folder, 
                        f'{mmcif_name.removesuffix(".cif")}.json'
                    ), 'wt'
                ) as f:
                f.write(alphafold_input.to_json())
            count += 1
        except:
            print(f'Converting {mmcif_name} failed.')
            continue
    print(f'Total converting {count}/{len(os.listdir(cif_folder))}')


def pdb2cif(input_pdb, output_cif):
    cmd.load(input_pdb, 'input_structure')
    cmd.save(output_cif, 'input_structure')
    cmd.delete('all')
    cmd.reinitialize()


def _process_cif_pymol(input_cif_path, output_cif_path):
    """
    使用 PyMOL 提取 CIF 文件中的 A 链和 B 链，剔除水分子，并保存为新的 CIF 文件。
    当初是为了批量处理那些包含多个蛋白多肽的结构，用于Benchmark
    参数：
        input_cif_path (str): 输入的 CIF 文件路径。
        output_cif_path (str): 输出的处理后 CIF 文件路径。
    """
    # 初始化 PyMOL 环境
    cmd.reinitialize()
    name = pathlib.Path(input_cif_path).stem
    # 加载 CIF 文件
    cmd.load(input_cif_path, name)

    # 获取所有链的标识符并提取前两条链
    chains = cmd.get_chains(name)
    if len(chains) >= 2:
        selected_chains = f"chain {chains[0]}+{chains[1]}"
    elif len(chains) == 1:
        selected_chains = f"chain {chains[0]}"
    else:
        raise ValueError("未找到任何链。")

    cmd.select('selected_chains', selected_chains)

    cmd.remove('resn HOH')
    cmd.remove('elem Cl+Zn')

    longest_chain = max(
                        cmd.get_chains('selected_chains'), 
                        key=lambda chain: cmd.count_atoms(f"chain {chain}")
                    )
    cmd.select('non_standard_longest', f'chain {longest_chain} and not polymer.protein')
    cmd.remove('non_standard_longest')

    cmd.save(output_cif_path, 'selected_chains')
    cmd.delete('all')
    cmd.reinitialize()


def process_cif_pymol_folder(input_cif_folder, output_cif_folder):
    
    count = 0
    for cif in tqdm(os.listdir(input_cif_folder)):
        input_cif_path = os.path.join(input_cif_folder, cif)
        output_cif_path = os.path.join(output_cif_folder, cif)
        try:
            _process_cif_pymol(input_cif_path, output_cif_path)
            count += 1
        except:
            print(f'Cleaning {cif} failed.')
            continue
    print(f'Total cleaning {count}/{len(os.listdir(input_cif_folder))}')