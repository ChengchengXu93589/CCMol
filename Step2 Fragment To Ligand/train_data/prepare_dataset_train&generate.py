import pandas as pd
import os
import re

# 文件的路径
input_file_path = './ChEMBL_BRICS_smiles.csv'  # 确保扩展名正确
output_file_path = './ChEMBL_BRICS_smiles_new.csv'

# 获取文件扩展名
extension = os.path.splitext(input_file_path)[1]


# 定义一个函数来替换[数字+*]格式
def clean_smiles(smiles):
    cleaned_smiles = re.sub(r"\[\d+\*\]", "[*]", smiles)
    return cleaned_smiles


if extension == '.csv':
    # 使用pandas读取CSV文件
    data = pd.read_csv(input_file_path)
    # 只处理'scaffold_smiles'列
    data['scaffold_smiles'] = data['scaffold_smiles'].apply(clean_smiles)

    # 保存修改后的DataFrame到新的CSV文件
    data.to_csv(output_file_path, index=False)

elif extension == '.smi':
    with open(input_file_path, 'r') as f:
        lines = f.readlines()

    # 应用clean_smiles函数到每一行
    cleaned_lines = [clean_smiles(line.strip()) for line in lines]

    # 将修改后的数据保存到新的文件
    with open(output_file_path, 'w') as f:
        f.write('\n'.join(cleaned_lines))