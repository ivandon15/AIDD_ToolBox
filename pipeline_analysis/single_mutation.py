import math
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def csv2dict(csv_file, aa_lst_idx, exp_col):
    """ 
    csv_file: columns [ID, AA1, AA2, ..., AAn, exp_value]
    aa_lst_idx: [x, y), refer the col index of AAs col
    exp_col: col name of exp value (usually the -log10KD)
    """
    df = pd.read_csv(csv_file)
    infos = {
        row.ID: {
            'exp_value': row[exp_col],
            'seq': list(row[aa_lst_idx])
        }
        for _, row in df.iterrows()
    }
    return infos


def find_one_difference_groups(infos):
    """
    find all the single mutation groups
    return: a list of tuple, (list of IDs, list of corresponding Seqs)
    """
    sequences = [v['seq'] for v in infos.values()]
    ids = list(infos.keys())

    def has_one_difference(seq1, seq2):
        return sum(a != b for a, b in zip(seq1, seq2)) == 1

    groups = []
    checked_pairs = set()  # 用于避免重复比较

    for (seq1, id1), (seq2, id2) in combinations(zip(sequences, ids), 2):
        if (id1, id2) in checked_pairs or (id2, id1) in checked_pairs:
            continue
        if has_one_difference(seq1, seq2):
            new_group = {(id1, tuple(seq1)), (id2, tuple(seq2))}
            for seq, id in zip(sequences, ids):
                if (id, tuple(seq)) not in new_group:
                    # 检查是否可以加入组
                    if all(has_one_difference(seq, member_seq) for _, member_seq in new_group):
                        new_group.add((id, tuple(seq)))
            # 转换为ID和序列列表
            new_group_ids = [id for id, _ in new_group]
            new_group_sequences = [list(seq) for _, seq in new_group]

            # 检查是否是子集
            if not any(set(new_group_ids).issubset(set(existing_group)) for existing_group, _ in groups):
                groups.append((new_group_ids, new_group_sequences))
            # 标记为已检查
            checked_pairs.update((id1, id2) for id1 in new_group_ids for id2 in new_group_ids)

    return groups


def analyze_groups(groups, infos):
    """
    analyze each group generated from find_one_difference_groups func
    return: {pos_idx: [(list of IDs, list of corresponding value, list of ID-AA)...,]}
    
    see the detailed outputs:
    for idx, group_info in idx_differences.items():
        print(f"Position {idx + 1}:")
        for group_ids, kd_values, differences in group_info:
            print(f"IDs: {group_ids}")
            print(f"Exp Values: {kd_values}")
            print(f"Details: {differences}")
    """
    idx_differences = defaultdict(list)

    for group_ids, group_seqs in groups:
        seq_len = len(group_seqs[0])
        for idx in range(seq_len):
            idx_group = [seq[idx] for seq in group_seqs]
            if len(set(idx_group)) > 1:
                differences = sorted((group_ids[i], group_seqs[i][idx]) for i in range(len(group_ids)))
                exp_values = [infos[id]['exp_value'] for id, _ in differences]
                idx_differences[idx].append((group_ids, exp_values, differences))

    return idx_differences


def visualize_heatmap(idx, group_info):
    """
    visualize the difference brought by single mutation
    """
    aa_pairs = []
    kd_deltas = []
    for _, kd_values, differences in group_info:
        for i, (_, aa1) in enumerate(differences):
            for j, (_, aa2) in enumerate(differences):
                if i < j:
                    aa_pairs.append((aa1, aa2))
                    if aa1 == aa2:
                        kd_delta = 0
                    elif not math.isnan(kd_values[i]) and not math.isnan(kd_values[j]):
                        kd_delta = kd_values[i] - kd_values[j]
                    else:
                        kd_delta = 0
                    kd_deltas.append(kd_delta)
    print("kd_deltas, ", kd_deltas)
    amino_acids = sorted(set([aa for pair in aa_pairs for aa in pair]))
    
    aa_index = {aa: i for i, aa in enumerate(amino_acids)}
    
    heatmap_data = np.zeros((len(amino_acids), len(amino_acids)))
    count_matrix = np.zeros((len(amino_acids), len(amino_acids)))
    
    for (aa1, aa2), kd_delta in zip(aa_pairs, kd_deltas):
        i, j = aa_index[aa1], aa_index[aa2]
        heatmap_data[i, j] += kd_delta
        count_matrix[i, j] += 1
    print("count_matrix, ", count_matrix)
    heatmap_data = np.divide(heatmap_data, count_matrix, where=count_matrix != 0)
    print("heatmap_data, ", heatmap_data)
    cmap = LinearSegmentedColormap.from_list('custom_diverging', ['#fc4e2a', '#ffffff', '#2ca25f'] , N=100)
    
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(heatmap_data, xticklabels=amino_acids, yticklabels=amino_acids, annot=False, cmap=cmap, center=0)

    for i in range(len(amino_acids)):
        for j in range(len(amino_acids)):
            value = heatmap_data[i, j]
            if abs(value) > 0.01 and (not math.isnan(value)) :
                ax.text(j + 0.5, i + 0.5, f'{value:.3f}', ha='center', va='center', color='black')
    
    plt.xlabel('Amino Acid')
    plt.ylabel('Amino Acid')
    plt.title(f'Amino Acid Changes and Avg Exp_value Delta at Position {idx + 1}\n from Y->X')
    plt.show()


csv_file = './241104_fcrn_wetlab_mono.csv'
exp_col = 'logKD'
aa_lst_idx = list(range(8,24))
infos = csv2dict(csv_file, aa_lst_idx, exp_col)
groups = find_one_difference_groups(infos)
idx_differences = analyze_groups(groups, infos)

idx_to_visualize = 1 # position
if idx_to_visualize in idx_differences:
    group_info = idx_differences[idx_to_visualize]
    visualize_heatmap(idx_to_visualize, group_info)
