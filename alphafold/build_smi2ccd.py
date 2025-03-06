import json
import re
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from other2afjson import canonical_smiles, ccd

def process_entry(code_v):
    """Process a single entry in the CCD dictionary."""
    code, v = code_v
    pattern = re.compile(r"[-+]|InChI=|\|")
    smi2ccd_local = {}
    
    if isinstance(v.get('_pdbx_chem_comp_descriptor.descriptor'), tuple):
        for smi in v['_pdbx_chem_comp_descriptor.descriptor']:
            if bool(pattern.search(smi)):
                continue
            try:
                canonical_smi = canonical_smiles(smi)
                smi2ccd_local[canonical_smi] = code
            except Exception as e:
                continue
    return smi2ccd_local

def build_smi2ccd_parallel(ccd):
    """Build a mapping from canonical SMILES to chemical component codes using multiprocessing."""
    smi2ccd = {}
    
    # 使用多进程池
    with Pool(processes=cpu_count()) as pool:
        # 使用 tqdm 显示进度条
        results = list(tqdm(pool.imap(process_entry, ccd.items()), total=len(ccd)))
    
    # 合并所有进程的结果
    for result in results:
        smi2ccd.update(result)
    
    return smi2ccd

# Build the smi2ccd mapping using multiprocessing
smi2ccd = build_smi2ccd_parallel(ccd)
# Save the result to a JSON file
json.dump(smi2ccd, open("smi2ccd.json", "w"))