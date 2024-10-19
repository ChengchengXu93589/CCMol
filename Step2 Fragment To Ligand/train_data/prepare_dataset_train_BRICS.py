from rdkit import Chem
from rdkit.Chem.BRICS import BRICSDecompose
import csv
import re

def clean_smiles(smiles):
    # Replace [number*] with [*] in the smiles string
    return re.sub(r"\[\d+\*\]", "[*]", smiles)

# Paths for input and output files
input_smi_path = 'ChEMBL.smi'
output_csv_path = 'ChEMBL_BRICS_smiles.csv'

# Open the input and output files
with open(input_smi_path, 'r') as input_file, open(output_csv_path, 'w', newline='') as csvfile:
    # Create a csv writer object for writing into the CSV file
    csvwriter = csv.writer(csvfile)
    # Write the CSV header
    csvwriter.writerow(['SMILES', 'scaffold_smiles'])

    for line in input_file:
        smiles = line.strip()  # Read a SMILES string from the file
        mol = Chem.MolFromSmiles(smiles)  # Convert the SMILES string to a molecule object
        if mol is not None:  # Check if the SMILES string is valid
            # Decompose the molecule using BRICS and get an iterable set of fragment SMILES
            fragment_smiles_set = BRICSDecompose(mol, returnMols=False, keepNonLeafNodes=False)
            # Iterate through the fragment SMILES set
            for fragment_smiles in fragment_smiles_set:
                if fragment_smiles:
                    cleaned_fragment = clean_smiles(fragment_smiles)  # Clean the fragment_smiles
                    # Write the original molecule and its cleaned fragment to the CSV file
                    csvwriter.writerow([smiles, cleaned_fragment])
        else:
            print(f"Invalid SMILES string: {smiles}")

print(f"Scaffold SMILES have been saved to {output_csv_path}")