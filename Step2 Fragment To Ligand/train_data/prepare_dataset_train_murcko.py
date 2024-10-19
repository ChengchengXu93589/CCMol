import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# 读取csv文件到DataFrame
input_csv_path = 'E:/MOL/zinc.csv'  # 输入csv文件路径
output_csv_path = 'E:/MOL/zinc_scaffold_smiles.csv'  # 输出csv文件路径
input_df = pd.read_csv(input_csv_path)

# 处理SMILES字符串，获取Murcko骨架，并创建新的DataFrame用于保存
scaffold_data = {'SMILES': [], 'scaffold_smiles': []}
for smiles in input_df['SMILES']:  # 假设原始SMILES字符串在名为'SMILES'的列中
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:  # 检查SMILES是否有效
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        scaffold_data['SMILES'].append(smiles)
        scaffold_data['scaffold_smiles'].append(scaffold_smiles)
    else:
        print(f"Invalid SMILES string: {smiles}")

# 将含有Murcko骨架的数据转换为DataFrame并保存到新的csv文件
scaffold_df = pd.DataFrame(scaffold_data)
scaffold_df.to_csv(output_csv_path, index=False)

print(f"Scaffold SMILES have been saved to {output_csv_path}")