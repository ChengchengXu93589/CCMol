from rdkit import Chem
from rdkit.Chem.Recap import RecapDecompose
import os

def process_smiles_file(input_file, output_file):
    # 打开输入和输出文件
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        seen_fragments = set()
        for line in infile:
            # 逐行读取SMILES字符串，并去掉换行符
            smiles = line.strip()
            if not smiles:
                continue

            # 将SMILES字符串转换为RDKit分子对象
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            # 使用RECAP分解获取片段树
            recap_tree = RecapDecompose(mol)
            if recap_tree is None:
                continue

            # 获取所有RECAP片段（不包括根节点）
            recap_fragments = recap_tree.GetLeaves()
            for frag in recap_fragments.values():
                frag_smiles = Chem.MolToSmiles(frag)
                seen_fragments.add(frag_smiles)

        # 将去重后的RECAP片段写入输出文件，每行一个
        for frag_smiles in seen_fragments:
            outfile.write(frag_smiles + '\n')

# 指定输入文件和输出文件
input_file = './input/JAK2.smi'
output_file = './output/JAK2_RECAP.smi'

# 调用函数处理文件
process_smiles_file(input_file, output_file)
