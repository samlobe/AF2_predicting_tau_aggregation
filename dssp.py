#%%
import mdtraj as md
import numpy as np
import pandas as pd
import pydssp
import matplotlib.pyplot as plt
import os
from glob import glob
from tqdm import tqdm

def get_hbonds(pdb_name,n_res,n_chains,plot=False):
    tot_residues = int(n_res*n_chains)

    # get chain id list (starting from A and continuing for n_chains e.g. to E)
    chains = [chr(i) for i in range(65,65+n_chains)]
    resids = np.arange(1,n_res+1)
    indices = [f'{chain}{resid}' for chain in chains for resid in resids]

    struct = md.load(pdb_name)
    # select just backbone atoms
    backbone_indices = struct.top.select('backbone')
    backbone = struct.atom_slice(backbone_indices)

    batch, all_atoms, xyz = backbone.xyz.shape
    coord = backbone.xyz.reshape(batch, int(all_atoms/4), 4, xyz)

    # create a mask of 15x15 squares on the diagonal to mask the intrachain hbonds
    intrachain_mask = np.zeros((tot_residues,tot_residues))
    startis = np.arange(0,tot_residues,n_res)
    for i in startis:
        intrachain_mask[i:n_res+i,i:n_res+i] = 1
    interchain_mask = 1 - intrachain_mask
    # make boolean
    intrachain_mask = intrachain_mask.astype(bool)
    interchain_mask = interchain_mask.astype(bool)
    if plot:
        plt.imshow(intrachain_mask); plt.title('Intrachain mask'); plt.show()
        plt.imshow(interchain_mask); plt.title('Interchain mask'); plt.show()

    hbond_matrix = pydssp.get_hbond_map(coord*10)[0] # converting to angstroms
    # select the interchain region of the hbond matrix
    interchain_hbonds = hbond_matrix[interchain_mask]
    num_interchain_hbonds = np.sum(interchain_hbonds)

    hbond_matrix = pd.DataFrame(hbond_matrix, columns=indices, index=indices)

    if plot:
        plt.imshow(hbond_matrix)
        # draw vertical lines every 15
        plt.axvline(x=14.5, color='w', linestyle='-', linewidth=1)
        plt.axvline(x=29.5, color='w', linestyle='-', linewidth=1)
        plt.axvline(x=44.5, color='w', linestyle='-', linewidth=1)
        plt.axvline(x=59.5, color='w', linestyle='-', linewidth=1)
        # draw horizontal lines every 15
        plt.axhline(y=14.5, color='w', linestyle='-', linewidth=1)
        plt.axhline(y=29.5, color='w', linestyle='-', linewidth=1)
        plt.axhline(y=44.5, color='w', linestyle='-', linewidth=1)
        plt.axhline(y=59.5, color='w', linestyle='-', linewidth=1)
        plt.title(f'Hbond matrix:\n{pdb_name}')
    return hbond_matrix, num_interchain_hbonds

# Define the base folders to search within (these are the types of AF2 approaches)
types = ['multimer', 'multimer_5rec', '5U', '10U']

# Define the models to search for, including a 'best' model which is rank 1
models = ['best', 'model_1', 'model_2', 'model_3', 'model_4', 'model_5']

# get the peptide names
tht_df = pd.read_csv('ThT_PAM4_paper.csv',index_col=0)
peptides = tht_df.columns[:-1] # ignoring control column

# initialize a nested dictionary to hold the paths of pdbs
files_dict = {type: {model: {} for model in models} for type in types}

# populate the dictionary with paths
for peptide in tqdm(peptides):
    for type in types:
        # Handle the best case
        best_pattern = f'{type}/{peptide}_unrelaxed_rank_001_*_seed_000.pdb'
        best_files = glob(best_pattern)
        if best_files:
            files_dict[type]['best'][peptide] = best_files[0]
        else:
            print(f"No 'best' file found for {peptide} in {type}.")

        # Handle the 5 models
        for model in models[1:]:
            pattern = f'{type}/{peptide}_*_{model}_seed_000.pdb'
            files = glob(pattern)
            if files:
                files_dict[type][model][peptide] = files[0]
            else:
                print(f"No file found for {peptide} with {model} in {type} directory.")

# make a distance matrix directory if it doesn't exist
for type in types:
    if not os.path.exists(f'hbonds/{type}'):
        os.makedirs(f'hbonds/{type}')

#%%
# Initialize a DataFrame to store the number of interchain hydrogen bonds
hbonds_summary = pd.DataFrame(columns=['Peptide', 'Type', 'Model', 'n_inter_hbonds'])

# Loop through all peptides, types, and models to calculate and store hbond data
for peptide in tqdm(peptides):
    for type in types:
        for model in models:
            path = files_dict[type][model].get(peptide)
            if not path or not os.path.exists(path):
                print(f"File not found or path is missing for {peptide}, {model} in {type}. Skipping...")
                continue
            
            try:
                hbond_df, n_inter_hbonds = get_hbonds(path, n_res=15, n_chains=5, plot=False)
                # Append the results to the summary DataFrame
                                # Create a temporary DataFrame for the current row of data
                temp_df = pd.DataFrame({
                    'Peptide': [peptide],
                    'Type': [type],
                    'Model': [model],
                    'n_inter_hbonds': [n_inter_hbonds]
                })
                # Concatenate the temporary DataFrame with the main DataFrame
                hbonds_summary = pd.concat([hbonds_summary, temp_df], ignore_index=True)
            except Exception as e:
                print(f"Error processing {path}: {e}")

# Save the summary of hydrogen bond counts to a CSV file
hbonds_summary.to_csv('hbonds/interchain_hbonds_summary.csv', index=False)
print("Saved hydrogen bond counts to 'interchain_hbonds_summary.csv'")

#%%
# # testing
# pdb_name = '10U/360-374_unrelaxed_rank_001_alphafold2_ptm_model_2_seed_000.pdb'
# n_res = 15
# n_chains = 5
# n_inter_hbonds =get_hbonds(pdb_name,n_res,n_chains,plot=True)
# print(n_inter_hbonds)

