#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import copy

types = ['multimer','5U','10U']

# get the peptide names
tht_df = pd.read_csv('ThT_PAM4_paper.csv',index_col=0)
peptides = tht_df.columns[:-1] # ignore the control column

# get the pAE paths for each peptide
pae_paths_multimer = {}
pae_paths_5U = {}
pae_paths_10U = {}

for peptide in peptides:
    # get the file paths
    pae_paths_multimer[peptide]= f'multimer/{peptide}_predicted_aligned_error_v1.json'
    pae_paths_5U[peptide] = f'5U/{peptide}_predicted_aligned_error_v1.json'
    pae_paths_10U[peptide] = f'10U/{peptide}_predicted_aligned_error_v1.json'

# put these paths in a dictionary
pae_paths = {'multimer': pae_paths_multimer, '5U': pae_paths_5U, '10U': pae_paths_10U}

# Define the indices and columns
chains = ['A', 'B', 'C', 'D', 'E']
resids = np.arange(1,16)
indices = [f'{chain}{resid}' for chain in chains for resid in resids]

# create a dictionary of the distance matrices for each type and peptide
contacts_multimer = {}
contacts_5U = {}
contacts_10U = {}
for type in types:
    distance_matrices_path = f'distance_matrices/{type}'
    # check if it exists and throw an error if it doesn't
    if not os.path.exists(distance_matrices_path):
        raise FileNotFoundError(f'{distance_matrices_path} not found')
    
    for peptide in peptides:
        for path in pae_paths[type]:
            if peptide in path:
                # read the distance matrix
                distance_matrix = pd.read_csv(f'{distance_matrices_path}/{peptide}.csv', header=None)
                distance_matrix.index = indices
                distance_matrix.columns = indices
                contact_mask = distance_matrix < 0.8 # using 0.8 nm as contact cutoff
                if type == 'multimer':
                    contacts_multimer[peptide] = contact_mask
                elif type == '5U':
                    contacts_5U[peptide] = contact_mask
                elif type == '10U':
                    contacts_10U[peptide] = contact_mask

# put these dataframes in a dictionary
contacts_dict = {'multimer': contacts_multimer, '5U': contacts_5U, '10U': contacts_10U}

#%%
# Function to remove intra-peptide contacts
def remove_intra_peptide_contacts(contact_mask):
    modified_mask = contact_mask.copy()
    n_residues_per_chain = 15
    n_chains = len(chains)
    for i in range(n_chains):
        start_idx = i * n_residues_per_chain
        end_idx = start_idx + n_residues_per_chain
        modified_mask.iloc[start_idx:end_idx, start_idx:end_idx] = False
    return modified_mask

# Apply the function to all contact masks
intercontacts_dict = {'multimer': {}, '5U': {}, '10U': {}}
for type_key, contacts in contacts_dict.items():
    for peptide, contact_mask in contacts.items():
        intercontacts_dict[type_key][peptide] = remove_intra_peptide_contacts(contact_mask)

#%%
def create_pae_dataframe(json_file_path):
    # Load the JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Extract the predicted aligned error matrix
    pae_matrix = data['predicted_aligned_error']

    type = json_file_path.split('/')[0]
    if type == '5U':
        # Calculate indices to delete
        del_indices = np.concatenate([np.arange(15,20), np.arange(35-5,40-5), np.arange(55-10,60-10),
                                    np.arange(75-15,80-15), np.arange(95-20,100-20)])
        # Delete rows and columns in one operation
        pae_matrix = np.delete(pae_matrix, del_indices, axis=0)
        pae_matrix = np.delete(pae_matrix, del_indices, axis=1)
    elif type == '10U':
        # Similar calculation for '10U'
        del_indices = np.concatenate([np.arange(15,25), np.arange(40-10,50-10), np.arange(65-20,75-20),
                                    np.arange(90-30,100-30), np.arange(115-40,125-40)])
        # Delete rows and columns in one operation
        pae_matrix = np.delete(pae_matrix, del_indices, axis=0)
        pae_matrix = np.delete(pae_matrix, del_indices, axis=1)

    
    # Define the indices and columns
    chains = ['A', 'B', 'C', 'D', 'E']
    resids = np.arange(1,16)
    indices = [f'{chain}{resid}' for chain in chains for resid in resids]
    
    # Create the DataFrame
    pae_df = pd.DataFrame(pae_matrix, index=indices, columns=indices)
    
    return pae_df

# %%
pae_dfs_dict = {'multimer': {}, '5U': {}, '10U': {}}
for type in types:
    for peptide in peptides:
        pae_dfs_dict[type][peptide] = create_pae_dataframe(pae_paths[type][peptide])

#%%
def plot_pAE_heatmap(df, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    
    # Create a copy of the 'viridis' colormap that sets 'bad' values (np.nan) to white
    cmap = plt.cm.bwr
    # cmap.set_bad(color='white')

    # Use the modified colormap for plotting
    im = ax.imshow(df, cmap=cmap, vmin=0, vmax=30)
    dim = df.shape[0]
    ax.set_xticks(np.arange(0, dim+1, 10))
    ax.set_yticks(np.arange(0, dim+1, 10))
    plt.xlabel('Residue Index')
    plt.ylabel('Residue Index')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='Predicted Aligned Error')
    plt.show()

plot_pAE_heatmap(pae_dfs_dict['multimer'][peptides[0]])

# %%
# zero out the pAE values for non-interchain contacts
# Mask the pAE dataframes based on interchain contacts
LIS_pae_dfs_dict = copy.deepcopy(pae_dfs_dict)
for type in types:
    for peptide in peptides:
        LIS_pae_dfs_dict[type][peptide] = LIS_pae_dfs_dict[type][peptide].mask(~intercontacts_dict[type][peptide])

plot_pAE_heatmap(LIS_pae_dfs_dict['multimer'][peptides[0]])

# %%
