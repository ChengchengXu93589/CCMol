import os
import random
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, FilterCatalog
from rdkit.Chem.rdMolDescriptors import CalcNumAromaticRings
from rdkit.Chem import Descriptors
from tqdm import tqdm
import subprocess
import meeko
import GA_mutate as mu
import GA_crossover as co
from GA_score import multi_scoring_functions_one_hot

# 定义警戒结构的SMARTS字符串
Insilico_alerts = [
    '[Cl,Br,F,I]c1ccncc1',  # para-substituted pyridines
    '[Cl,Br,F,I]c1ccccn1',  # ortho halogen-substituted pyridines
    '[Cl,Br,F,I]C1=CC=CO1',
    '[Cl,Br,F,I]C1=CC=CS1',
    '[#16][#16]',  # disulfide
    'O=C1Nc2ccccc2C1=O',  # isatin
    'O=C1CC(=O)NC(=[O,S])N1',  # barbiturates
    '[A][$([#6]=[N+]=[N-]),$([#6]-[N+]#[N])]',  # diazo-compounds
    '[OX2,OX1-][OX2,OX1-]',  # peroxides
    '[OX2][$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]',  # sulfonyl ester
    '[I]',  # iodine
    '[Si]',  # silicon
    '[Ge]',  # germanium
    '[Sn]',  # tin
    '[Pb]',  # lead
    'C#C',  # alkyne
    '[N;X3;v3,v5+0]([O,N])[O,N]',  # nitro
    'c1ccccc1N(=O)=O',  # nitroaromatic
    'C1=NN=NN1',  # tetrazole
    '[N,O,S]-N=[N+]=[N-]',  # azide
    '[O]=C=O',  # carbon dioxide
    '[C]=S',  # thiocarbonyl
    '[#3,#11,#19,#37,#55,#87]',  # metal element
    '[R]1[R][R][R]1',  # 三环结构
    'S(=O)(=O)[O-]',  # 磺酸基团
    '[O,S]@[O,S]',  # 同一个环中同时含有氧硫两种原子
    '[N,S]@[N,S]',  # 同一个环中同时含有氮硫两种原子
    '[O,N]@[O,N]',  # 同一个环内同时含有氮氧两种原子
    'C1=C-C-C=C-C1',  # 孤立的六元环中只有一个双键
    'C1=CC=CC=C1',  # 孤立的六元环中有两个双键
    'CCCC',  # 超过四个碳原子的长链结构
    'C-S-C',  # 硫作为取代原子取代碳原子
    'OS',  # 硫氧单键
    '[N]=[N]',  # 末端氮不饱和键结构（=NH）
    '[C]=[C]',  # 末端双键结构（=烯基结构）
    'N1CCCCC1',  # 同一个环中含有多个氮原子且该环没有芳香性
    '[R]1[R]@[R]2[R]1[R]2',  # 螺环连接的小环结构（比如五元或六元交三元螺环）
    'C=N',  # 碳氮双键
    'C(C)(F)F',  # 三氟甲基被取代的结构
    'C(C)(C)F',  # 三氟甲基被取代的结构
    'N-O',  # 氮上连有羟基
]

# 定义基团出现的频率上限
count_alert = {'Cl': 3, 'F': 6, 'Br': 2, '[N+](=O)[O-]': 2}

# 转换警戒结构为RDKit分子对象
alert_mols = []
for smi in Insilico_alerts:
    mol = Chem.MolFromSmarts(smi)
    if mol is None:
        print(f"Error creating molecule from SMARTS: {smi}")
    alert_mols.append(mol)


def check_alerts(mol, alert_mol_list=alert_mols, alert_dict=count_alert):
    for query in alert_mol_list:
        if query is None:
            continue  # 跳过无效的警戒结构
        if mol.HasSubstructMatch(query):
            return False
    for alert in alert_dict.keys():
        query = Chem.MolFromSmarts(alert)
        if len(mol.GetSubstructMatches(query)) > alert_dict[alert]:
            return False
    if CalcNumAromaticRings(mol) >= 6:
        return False
    return True


# PAINS filter setup
param = FilterCatalog.FilterCatalogParams()
param.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
pains_filter = FilterCatalog.FilterCatalog(param)


def check_lipinski(mol):
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)

    if mw > 500:
        return False
    if logp > 5:
        return False
    if hbd > 5:
        return False
    if hba > 10:
        return False
    if rot_bonds > 10:
        return False

    return True


def is_valid_molecule(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return False

    # Check for valency errors
    for atom in mol.GetAtoms():
        if atom.GetExplicitValence() > atom.GetTotalValence():
            return False

    # Check aromaticity consistency
    for bond in mol.GetBonds():
        if bond.GetIsAromatic() != (bond.GetBondType() == Chem.rdchem.BondType.AROMATIC):
            return False

    return True


def check_pains(mol):
    return not pains_filter.HasMatch(mol)


def is_acceptable_molecule(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return False

    # Check for valency errors
    for atom in mol.GetAtoms():
        if atom.GetExplicitValence() > atom.GetTotalValence():
            return False

    # Check aromaticity consistency
    for bond in mol.GetBonds():
        if bond.GetIsAromatic() != (bond.GetBondType() == Chem.rdchem.BondType.AROMATIC):
            return False

    return check_pains(mol) and check_alerts(mol) and check_lipinski(mol)


def read_smi_file(file_name):
    mol_list = []
    with open(file_name, 'r') as file:
        for line in file:
            smiles = line.strip()
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol_list.append(mol)
    return mol_list


def make_initial_population(population_size, file1, file2):
    mol_list1 = read_smi_file(file1)
    mol_list2 = read_smi_file(file2)
    combined_list = mol_list1 + mol_list2
    population = []
    while len(population) < population_size:
        new_mol = random.choice(combined_list)
        if is_acceptable_molecule(new_mol):  # 确保分子符合所有过滤标准
            new_smiles = Chem.MolToSmiles(new_mol)
            if new_smiles not in [Chem.MolToSmiles(mol) for mol in population]:  # 确保种群中没有重复的分子
                population.append(new_mol)
    return population


def calculate_normalized_fitness(population):
    scores = multi_scoring_functions_one_hot(population,
                                             ['qed', 'sa', 'logP', 'TPSA', 'MW', 'HBD', 'HBA', 'RotatableBonds'])
    fitness = scores
    sum_fitness = sum(fitness)
    normalized_fitness = [fit_idv / sum_fitness for fit_idv in fitness]
    return normalized_fitness


def make_mating_pool(population, fitness, population_size):
    mating_pool = []
    selected = set()
    while len(mating_pool) < population_size:
        selected_mol = np.random.choice(population, p=fitness)
        selected_smiles = Chem.MolToSmiles(selected_mol)
        if selected_smiles not in selected:
            selected.add(selected_smiles)
            mating_pool.append(selected_mol)
    return mating_pool


def reproduce(mating_pool, population_size, mutation_rate):
    new_population = []
    generated_smiles = set()
    while len(new_population) < population_size:
        parent_A = random.choice(mating_pool)
        parent_B = random.choice(mating_pool)
        new_child = co.crossover(parent_A, parent_B)
        if new_child is not None:
            mutated_child = mu.mutate(new_child, mutation_rate)
            if mutated_child is not None and is_acceptable_molecule(mutated_child):
                new_smiles = Chem.MolToSmiles(mutated_child)
                if new_smiles not in generated_smiles:
                    generated_smiles.add(new_smiles)
                    new_population.append(mutated_child)
    print(f'Generated new population size: {len(new_population)} / {population_size}')
    return new_population


def vina_docking(new_population, vina, protein_file, config_file, docking_dir, pdb_id, generation):
    generation_dir = os.path.join(docking_dir, f"generation_{generation}")
    ligand_dir = os.path.join(generation_dir, "ligands")
    os.makedirs(ligand_dir, exist_ok=True)
    ligand_files = []
    for idx, mol in enumerate(new_population):
        smi = Chem.MolToSmiles(mol)
        lig = Chem.MolFromSmiles(smi)
        protonated_lig = Chem.AddHs(lig)
        if AllChem.EmbedMolecule(protonated_lig) == -1:
            print(f"Embedding failed for SMILES string: {smi}")
            continue
        meeko_prep = meeko.MoleculePreparation()
        meeko_prep.prepare(protonated_lig)
        ligand_file = os.path.join(ligand_dir, f"ligand_{idx}.pdbqt")
        with open(ligand_file, "w") as f:
            f.write(meeko_prep.write_pdbqt_string())
        ligand_files.append((ligand_file, smi))

    print("Starting docking process...")
    scores = []
    for idx, (ligand_file, smi) in enumerate(tqdm(ligand_files, desc="Docking ligands")):
        output_file = os.path.join(generation_dir, f"{pdb_id}_ligand_{idx}_vina_out.pdbqt")
        log_file = os.path.join(generation_dir, f"{pdb_id}_ligand_{idx}_vina_out.log")
        command = [vina, "--receptor", protein_file, "--ligand", ligand_file, "--config", config_file,
                   "--exhaustiveness", "32", "--out", output_file]
        with open(log_file, "w") as f:
            subprocess.run(command, stdout=f)
        ligand_scores = []
        with open(log_file, "r") as f:
            lines = f.readlines()
            start = False
            for line in lines:
                if line.strip() == "-----+------------+----------+----------":
                    start = True
                elif start:
                    try:
                        score = float(line.split()[1])
                        ligand_scores.append(score)
                    except (ValueError, IndexError):
                        continue
        if ligand_scores:
            average_score = sum(ligand_scores) / len(ligand_scores)
            scores.append((average_score, smi, new_population[idx]))

    scores.sort()
    print(f"Docking completed. {len(scores)} scores obtained.")
    return scores


def tanimoto_similarity(mol1, mol2):
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


# 遗传算法主流程
population_size = 250
mutation_rate = 0.05
file1 = '../Step2 Fragment To Ligand/generated_data/JAK2_ligand_generation.smi'
file2 = '../Step3 Structure To Ligand/structure_output/JAK2_structure_generation.smi'
similarity_dataset_file = './JAK2.smi'

vina = './autodock_vina_1_1_2_linux_x86/bin/vina'
pdb_id = "7Q7K"  # 请替换为你的PDB ID
protein_file = f"{pdb_id}.pdbqt"
config_file = f"{pdb_id}_receptor_vina_box.txt"

# 创建docking文件夹
docking_dir = "./JAK2_docking"
os.makedirs(docking_dir, exist_ok=True)

# 创建output文件夹
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

final_results = set()
max_final_results_size = 10000  # 设定最终结果的数量

# 初始种群生成
print("Generating initial population...")
population = make_initial_population(population_size, file1, file2)
similarity_dataset = read_smi_file(similarity_dataset_file)
print(f"Initial population generated with size: {population_size}")

generation = 0

while len(final_results) < max_final_results_size:
    print("Starting new round of evolution...")

    # Vina对接打分
    scores = vina_docking(population, vina, protein_file, config_file, docking_dir, pdb_id, generation)

    # 选择打分较好的分子
    top_solutions = [score[2] for score in scores if score[0] < -7]

    # 更新最终结果集
    with open(os.path.join(output_dir, 'JAK2_ccmol0821.smi'), 'a') as final_results_file:
        for mol in top_solutions:
            if is_acceptable_molecule(mol):  # 确保分子符合所有过滤标准
                smiles = Chem.MolToSmiles(mol)
                if smiles not in final_results:
                    final_results.add(smiles)
                    final_results_file.write(smiles + '\n')

    print(f"Top solutions: {len(top_solutions)}, Final results: {len(final_results)}")

    # 如果最终结果集达到了设定的数量，跳出循环
    if len(final_results) >= max_final_results_size:
        break

    # 对打分较好的分子进行相似性搜索
    similarity_scores = []
    for mol in tqdm(top_solutions, desc="Calculating similarities for top solutions"):
        max_similarity = 0
        for ref_mol in similarity_dataset:
            similarity = tanimoto_similarity(mol, ref_mol)
            if similarity > max_similarity:
                max_similarity = similarity
        similarity_scores.append((max_similarity, mol))

    # 选择相似性最高的前50%的分子作为下一代种群的一部分
    similarity_scores.sort(reverse=True, key=lambda x: x[0])
    top_50_percent_index = int(0.5 * len(similarity_scores))
    next_generation = [score[1] for score in similarity_scores[:top_50_percent_index]]

    # 使用相似性搜索选出的分子进行下一代遗传算法
    fitness = calculate_normalized_fitness(next_generation)
    mating_pool = make_mating_pool(next_generation, fitness, len(next_generation))
    new_population = reproduce(mating_pool, population_size, mutation_rate)  # 生成新的分子

    # 更新种群
    population = new_population
    print(f"New population generated with size: {len(population)}")

    generation += 1

print("Final results saved.")