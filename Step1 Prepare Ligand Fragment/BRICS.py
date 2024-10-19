import rdkit
from rdkit import Chem
from rdkit.Chem import BRICS
import os
import re

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

            # 进行BRICS碎片化
            fragments = BRICS.BRICSDecompose(mol)

            # 格式化碎片，去除断点位置的数字信息，只保留[*]
            for frag in fragments:
                formatted_frag = re.sub(r"\[\d+\*\]", "", frag)
                seen_fragments.add(formatted_frag)

        # 将去重后的碎片写入输出文件，每行一个碎片
        for formatted_frag in seen_fragments:
            outfile.write(formatted_frag + '\n')

# 指定输入文件和输出文件
input_file = './input/WRN.smi'
output_file = './output/WRN_BRICS.smi'

# 调用函数处理文件
process_smiles_file(input_file, output_file)
