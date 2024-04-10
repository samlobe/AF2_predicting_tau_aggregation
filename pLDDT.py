#%%
import pandas as pd
import numpy as np
import os

types = ['multimer','5U','10U']

# get the peptide names
tht_df = pd.read_csv('ThT_PAM4_paper.csv',index_col=0)
peptides = tht_df.columns[:-1] # ignoring the control column

# get the file paths for each peptide
# it should begin with '{peptide}_unrelaxed_rank_001' and end with 'seed_000.pdb'
paths_multimer = []
paths_5U = []
paths_10U = []

for peptide in peptides:
    # starts with
    start = f'{peptide}_unrelaxed_rank_001'
    # ends with
    end = 'seed_000.pdb'
    # get the file path
    for type in types:
        for file in os.listdir(type):
            if file.startswith(start) and file.endswith(end):
                if type == 'multimer':
                    paths_multimer.append(f'{type}/{file}')
                elif type == '5U':
                    paths_5U.append(f'{type}/{file}')
                elif type == '10U':
                    paths_10U.append(f'{type}/{file}')

# Define a function to parse PDB files for pLDDT scores
def parse_plddt_from_pdb(filepath):
    plddt_scores = []
    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith('ATOM') and line[13:15].strip() == 'C':
                # Assuming pLDDT score is stored in the B-factor column, typically found from characters 61 to 66 in PDB format
                # Adjust the indices if necessary based on the PDB file format
                plddt_score = float(line[60:66].strip())
                plddt_scores.append(plddt_score)
    return np.mean(plddt_scores)

# Initialize dictionaries to store the pLDDT scores
plddt_scores = {'multimer': [], '5U': [], '10U': []}

# Define the paths
types_paths = {'multimer': paths_multimer, '5U': paths_5U, '10U': paths_10U}

# Loop through each type and each file to extract pLDDT scores
for type, paths in types_paths.items():
    for path in paths:
        plddt_scores[type].append(parse_plddt_from_pdb(path))

#%%
# put in a dataframe
df = pd.DataFrame(plddt_scores)
# set the index to the peptide names
df.index = peptides
# name the index column 'peptide'
df.index.name = 'peptide'
# %%

# plot the pLDDT scores as a bar plot
import matplotlib.pyplot as plt
df.plot(kind='bar',figsize=(10,5))
plt.ylabel('pLDDT score')
plt.xlabel('Peptide')