import torch
from transformers import AutoTokenizer, OpenAIGPTLMHeadModel
from rdkit import Chem
from tqdm import tqdm
import os

# 加载标记器和模型
tokenizer = AutoTokenizer.from_pretrained('DeepChem/SmilesTokenizer_PubChem_1M')
model = OpenAIGPTLMHeadModel.from_pretrained('./checkpoint_epoch_3')

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# 生成分子
def generate_molecule(scaffold_smiles, num_molecules=10, num_beams=20, temperature=1.0, top_k=50, top_p=0.9, repetition_penalty=1.0):
    if num_molecules > num_beams:
        raise ValueError(f"`num_return_sequences` ({num_molecules}) has to be smaller or equal to `num_beams` ({num_beams}).")
    inputs = tokenizer(scaffold_smiles, return_tensors='pt').to(device)
    outputs = model.generate(
        **inputs,
        max_length=100,
        num_beams=num_beams,
        num_return_sequences=num_molecules,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True  # 启用采样模式
    )
    generated_smiles_list = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    torch.cuda.empty_cache()
    return generated_smiles_list

# 验证生成分子有效性
def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

# 从SMI文件读取骨架碎片
def read_scaffold_smiles(file_path):
    with open(file_path, 'r') as file:
        scaffolds = [line.strip() for line in file.readlines()]
    return scaffolds

# 保存生成的分子到SMI文件
def save_generated_molecule(file_path, smiles):
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a') as file:  # 使用 'a' 模式追加写入
        file.write(f"{smiles}\n")

# 示例生成和保存
smi_file_path = '../Step1 Prepare Ligand Fragment/output/JAK2_Murcko.smi'
scaffolds = read_scaffold_smiles(smi_file_path)
generated_molecules = set()

# 生成分子的数量
num_molecules_per_scaffold = 150  # 增加生成分子的数量
num_beams = 150  # 增加束搜索的束数以增加多样性
temperature = 1.5  # 增加温度以增加多样性
top_k = 100  # 增加 top_k 以增加多样性
top_p = 0.98  # 增加 top_p 以增加多样性
repetition_penalty = 1.8  # 使用重复惩罚以减少重复

output_smi_file_path = './generated_data/JAK2_ligand_generation.smi'  # 指定输出文件路径

for scaffold in tqdm(scaffolds, desc="Generating molecules"):
    generated_smiles_list = generate_molecule(
        scaffold,
        num_molecules=num_molecules_per_scaffold,
        num_beams=num_beams,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    for smiles in generated_smiles_list:
        if is_valid_smiles(smiles) and smiles not in generated_molecules:
            generated_molecules.add(smiles)
            save_generated_molecule(output_smi_file_path, smiles)  # 生成一个分子就写入一次

print(f"Generated molecules saved to {output_smi_file_path}")
