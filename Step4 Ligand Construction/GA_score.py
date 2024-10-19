import numpy as np
import pandas as pd
from rdkit import Chem
import rdkit.Chem.QED as QED
from rdkit.Chem import Descriptors as Des
from rdkit.Contrib.SA_Score import sascorer

class qed_func():
    def __call__(self, mol_list):
        scores = []
        for mol in mol_list:
            if mol is None:
                scores.append(0)
            else:
                try:
                    scores.append(QED.qed(mol))
                except Exception as e:
                    scores.append(0)
        return np.float32(scores)

class sa_func():
    def __call__(self, mol_list):
        scores = []
        for mol in mol_list:
            if mol is None:
                scores.append(100)
            else:
                try:
                    scores.append(sascorer.calculateScore(mol))
                except Exception as e:
                    scores.append(100)
        return np.float32(scores)

class logP_func():
    def __call__(self, mol_list):
        scores = []
        for mol in mol_list:
            if mol is None:
                scores.append(100)
            else:
                try:
                    scores.append(Des.MolLogP(mol))
                except Exception as e:
                    scores.append(100)
        return np.float32(scores)

class TPSA_func():
    def __call__(self, mol_list):
        scores = []
        for mol in mol_list:
            if mol is None:
                scores.append(0)
            else:
                try:
                    scores.append(Des.TPSA(mol))
                except Exception as e:
                    scores.append(0)
        return np.float32(scores)

class MW_func():
    def __call__(self, mol_list):
        scores = []
        for mol in mol_list:
            if mol is None:
                scores.append(0)
            else:
                try:
                    scores.append(Des.MolWt(mol))
                except Exception as e:
                    scores.append(0)
        return np.float32(scores)

class HBD_func():
    def __call__(self, mol_list):
        scores = []
        for mol in mol_list:
            if mol is None:
                scores.append(0)
            else:
                try:
                    scores.append(Des.NumHDonors(mol))
                except Exception as e:
                    scores.append(0)
        return np.float32(scores)

class HBA_func():
    def __call__(self, mol_list):
        scores = []
        for mol in mol_list:
            if mol is None:
                scores.append(0)
            else:
                try:
                    scores.append(Des.NumHAcceptors(mol))
                except Exception as e:
                    scores.append(0)
        return np.float32(scores)

class RotatableBonds_func():
    def __call__(self, mol_list):
        scores = []
        for mol in mol_list:
            if mol is None:
                scores.append(0)
            else:
                try:
                    scores.append(Des.NumRotatableBonds(mol))
                except Exception as e:
                    scores.append(0)
        return np.float32(scores)

def get_scoring_function(prop_name):
    """Function that initializes and returns a scoring function by name"""
    if prop_name == 'qed':
        return qed_func()
    elif prop_name == 'sa':
        return sa_func()
    elif prop_name == 'logP':
        return logP_func()
    elif prop_name == 'TPSA':
        return TPSA_func()
    elif prop_name == 'MW':
        return MW_func()
    elif prop_name == 'HBD':
        return HBD_func()
    elif prop_name == 'HBA':
        return HBA_func()
    elif prop_name == 'RotatableBonds':
        return RotatableBonds_func()
    else:
        return None

def multi_scoring_functions_one_hot(data, function_list):
    funcs = [get_scoring_function(prop) for prop in function_list]
    props = np.array([func(data) for func in funcs]).T

    props = pd.DataFrame(props)
    props.columns = function_list
    #print('props before condition_convert:', props)

    scoring_sum = condition_convert(props).values.sum(1)
    #print('scoring_sum after condition_convert:', scoring_sum)
    # scoring_sum = props.sum(axis=0)

    return scoring_sum

def condition_convert(con_df):
    # convert to 0, 1
    con_df['qed'][con_df['qed'] >= 0.6] = 1
    con_df['qed'][con_df['qed'] < 0.6] = 0
    con_df['sa'][con_df['sa'] <= 4.0] = 1
    con_df['sa'][con_df['sa'] > 4.0] = 0
    con_df['logP'][(con_df['logP'] >= -2.0) & (con_df['logP'] <= 6.0)] = 1
    con_df['logP'][(con_df['logP'] > 6.0) | (con_df['logP'] < -2.0)] = 0
    con_df['TPSA'][(con_df['TPSA'] >= 20.0) & (con_df['TPSA'] <= 140.0)] = 1
    con_df['TPSA'][(con_df['TPSA'] > 140.0) | (con_df['TPSA'] < 20.0)] = 0
    con_df['MW'][(con_df['MW'] >= 250) & (con_df['MW'] <= 450)] = 1
    con_df['MW'][(con_df['MW'] > 450) | (con_df['MW'] < 250)] = 0
    con_df['HBD'][con_df['HBD'] < 5] = 1
    con_df['HBD'][con_df['HBD'] >= 5] = 0
    con_df['HBA'][con_df['HBA'] < 10] = 1
    con_df['HBA'][con_df['HBA'] >= 10] = 0
    con_df['RotatableBonds'][con_df['RotatableBonds'] < 10] = 1
    con_df['RotatableBonds'][con_df['RotatableBonds'] >= 10] = 0

    #print('con_df after condition_convert:', con_df)

    return con_df

if __name__ == "__main__":
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--prop', required=True)

    args = parser.parse_args()
    funcs = [get_scoring_function(prop) for prop in args.prop.split(',')]

    data = [line.split()[:2] for line in sys.stdin]
    all_x, all_y = zip(*data)
    props = [func(all_y) for func in funcs]

    col_list = [all_x, all_y] + props
    for tup in zip(*col_list):
        print(*tup)