import random
import math
import hashlib
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import MolFromPDBFile
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import QED
from rdkit.Chem import Descriptors
from rdkit.Chem.BRICS import BRICSDecompose
import numpy as np
import pandas as pd
import os
from openbabel.pybel import *
import time
import multiprocessing, traceback
from scipy.spatial.distance import cdist
from func_timeout import func_set_timeout

import GA_mutate as mu
import GA_crossover as co
from GA_score import multi_scoring_functions_one_hot
from rdkit.Contrib.SA_Score import sascorer as sc
from tqdm import tqdm


# 输入文件夹路径和输出CSV文件路径
folder_path = 'E:/MOL/record'
output_csv = 'E:/MOL/record.csv'

# 创建空的DataFrame来存储SMILES
df = pd.DataFrame(columns=['SMILES'])

# 遍历文件夹中的SDF文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.sdf'):
        file_path = os.path.join(folder_path, file_name)

        # 使用RDKit读取SDF文件
        supplier = Chem.SDMolSupplier(file_path)

        # 提取有效的SMILES并添加到DataFrame
        for molecule in supplier:
            if molecule is not None:
                smiles = Chem.MolToSmiles(molecule, isomericSmiles=True)
                df = df.append({'SMILES': smiles}, ignore_index=True)
            else:
                print(f'Invalid molecule in file: {file_name}')

# 将DataFrame保存为CSV文件
df.to_csv(output_csv, index=False)


def read_file(file_name):
    mol_list = []
    with open(file_name,'r') as file:
      for smiles in file:
       mol_list.append(Chem.MolFromSmiles(smiles))

    return mol_list

def decompose_molecules(mol_list):
    decomposed_fragments = set()
    for mol in mol_list:
        if mol is not None:
            fragments = BRICSDecompose(mol)
            decomposed_fragments.update(fragments)
    return list(decomposed_fragments)


def make_initial_population(population_size, file_name):
    """
    Generate the initial population for the genetic algorithm from decomposed fragments.
    """
    mol_list = read_file(file_name)
    # 拆分分子为碎片
    decomposed_fragments = decompose_molecules(mol_list)
    # 从碎片中选择初始种群
    population = []
    for _ in range(population_size):
        random_fragment_smiles = random.choice(decomposed_fragments)
        mol = Chem.MolFromSmiles(random_fragment_smiles)
        if mol:  # 确保从SMILES字符串成功创建了分子
            population.append(mol)
    return population


def calculate_normalized_fitness(score):
    # Scoring_function采用均值
    score = multi_scoring_functions_one_hot(population, ['qed', 'sa', 'logP', 'TPSA'])
    #print('score:', score)

    fitness = score

    sum_fitness = sum(fitness)
    normalized_fitness = [fit_idv / sum_fitness for fit_idv in fitness]
    return normalized_fitness

def make_mating_pool(population, fitness):
    mating_pool = []
    for i in range(population_size):
        mating_pool.append(np.random.choice(population, p=fitness, replace=False))

    return mating_pool

def reproduce(mating_pool,population_size,mutation_rate):
  new_population = []
  while len(new_population) < population_size:
    parent_A = random.choice(mating_pool)
    parent_B = random.choice(mating_pool)
    new_child = co.crossover(parent_A,parent_B)
    #print(parent_A)
    #print(parent_B)
    if new_child != None:
      mutated_child = mu.mutate(new_child,mutation_rate)
      if mutated_child != None:
        #print(','.join([Chem.MolToSmiles(mutated_child),Chem.MolToSmiles(new_child),Chem.MolToSmiles(parent_A),Chem.MolToSmiles(parent_B)]))
        new_population.append(mutated_child)
  #print(new_population)
  print('length of new_population:' ,len(new_population))
  return new_population

population_size = 100
generations = 10
mutation_rate = 0.01
file_name = 'record.csv'

t0 = time.time()
all_active_list = []
for i in tqdm(range(10), desc="Main Loop"):
    population = make_initial_population(population_size, file_name)
    score = multi_scoring_functions_one_hot(population, ['qed', 'sa', 'logP', 'TPSA'])

    for generation in tqdm(range(generations), desc=f"Generation {i}"):
        fitness = calculate_normalized_fitness(score)
        mating_pool = make_mating_pool(population, fitness)
        new_population = reproduce(mating_pool, population_size, mutation_rate)
        new_population_smi = []
        for mol in new_population:
            smi = Chem.MolToSmiles(mol)
            new_population_smi.append(smi)
        #print(new_population_smi)
        # 每一步都写
        every_active = pd.DataFrame(new_population_smi)
        every_active.to_csv('./output/first_model_no_canonical_large.smi', index=False, header=False, mode='a')
        all_active_list.extend(new_population)
        print('len of set: ', len(set(all_active_list)))

    if len(set(all_active_list)) > 10000:
        all_active_list = pd.DataFrame(set(all_active_list))
        all_active_list.to_csv('./output/first_model_no_canonical_large.smi', index=False, header=False)
        break

t1 = time.time()
print('time ', t1 - t0)
