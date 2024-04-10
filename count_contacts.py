#%%
import numpy as np
import pandas as pd
import mdtraj as md
import os
from itertools import combinations
from matplotlib import pyplot as plt
from tqdm import tqdm

def get_pairdistances(peptide_path):
    AF2_pdb = md.load(peptide_path)
    topology = AF2_pdb.topology

    alpha_carbons = ([atom.index for atom in topology.atoms if atom.name == 'CA'])
    atom_pairs = list(combinations(alpha_carbons,2))
    pairwise_distances = md.geometry.compute_distances(AF2_pdb, atom_pairs)[0]

    num_residues = AF2_pdb.n_residues  # Number of residues in the protein
    atom_to_residue = {atom.index: atom.residue.index for atom in topology.atoms if atom.name == 'CA'}

    # Initialize an empty 2D matrix for the distances
    distance_matrix = np.zeros((num_residues, num_residues))

    # Fill the distance matrix. Since the distances are symmetric, we mirror the values across the diagonal
    for distance, (atom_index_i, atom_index_j) in zip(pairwise_distances, atom_pairs):
        residue_index_i = atom_to_residue[atom_index_i]
        residue_index_j = atom_to_residue[atom_index_j]
        
        # Populate the matrix, adjusting indices by -1 if necessary
        # Adjust the indices based on how your residues are indexed (0-based or 1-based)
        distance_matrix[residue_index_i][residue_index_j] = distance
        distance_matrix[residue_index_j][residue_index_i] = distance  # Mirror the distance for symmetry

    return distance_matrix

def get_chain_info(peptide_path):
    AF2_pdb = md.load(peptide_path)
    topology = AF2_pdb.topology
    
    if peptide_path.split("/")[0] == 'multimer':
        chain_info = []
        for chain in topology.chains:
            chain_residues = [residue.index for residue in chain.residues]
            chain_start = min(chain_residues)
            chain_end = max(chain_residues)
            chain_info.append((chain_start, chain_end))
    # hard coded chain info for 5U and 10U since every residue is chain A
    else: chain_info = [(0, 14), (15, 29), (30, 44), (45, 59), (60, 74)]

    return chain_info

def count_interchain_contacts(distance_matrix, peptide_path):
    contact_map = distance_matrix < 0.8  # Assuming <0.8 nm as contact threshold
    
    interchain_contacts = 0
    chain_info = get_chain_info(peptide_path)
    for i, (start_i, end_i) in enumerate(chain_info):
        for j, (start_j, end_j) in enumerate(chain_info):
            if i < j:  # Avoid double counting and self-counting
                # Slice the contact map to only consider contacts between chains i and j
                interchain_section = contact_map[start_i:end_i+1, start_j:end_j+1]
                interchain_contacts += np.sum(interchain_section)
    
    return interchain_contacts

def plot_pairdistances(distance_matrix, peptide_path):
    peptide_name = peptide_path.split("/")[1].split("_")[0]
    AF2_pdb = md.load(peptide_path)
    topology = AF2_pdb.topology
    chain_info = get_chain_info(peptide_path)
    chain_starts = [start for start, _ in chain_info]

    plt.figure(figsize=(10, 8))
    plt.imshow(distance_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Distance (nm)')
    
    # Draw lines to demarcate different chains
    for start in chain_starts:
        print(start)
        plt.axvline(x=start-0.5, color='black')
        plt.axhline(y=start-0.5, color='black')

    plt.title(f'Pairwise Distance Heatmap: {peptide_name}')
    plt.xlabel('Residue Index')
    plt.ylabel('Residue Index')
    plt.show()


def get_contactmap(distance_matrix, peptide_path):
    peptide_name = peptide_path.split("/")[1].split("_")[0]

    # output mask of contacts (<0.6nm)
    contact_map = distance_matrix < 0.6
    chain_info = get_chain_info(peptide_path)
    chain_starts = [start for start, _ in chain_info]

    plt.figure(figsize=(10, 8))
    plt.imshow(contact_map, cmap='viridis', interpolation='nearest')
    
    # Draw lines to demarcate different chains
    for start in chain_starts:
        plt.axvline(x=start-0.5, color='black')
        plt.axhline(y=start-0.5, color='black')

    plt.title(f'Contact Map: {peptide_name}')
    plt.xlabel('Residue Index')
    plt.ylabel('Residue Index')
    plt.show()

types = ['multimer','5U','10U']

# get the peptide names
tht_df = pd.read_csv('ThT_PAM4_paper.csv',index_col=0)
peptides = tht_df.columns 

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

#%%
# make a distance matrix directory if it doesn't exist

for type in types:
    if not os.path.exists(f'distance_matrices/{type}'):
        os.makedirs(f'distance_matrices/{type}')

interchain_contacts = {}
plot_flag = False
for type in types:
    if type == 'multimer': paths = paths_multimer
    elif type == '5U':     paths = paths_5U
    elif type == '10U':    paths = paths_10U

    count = 0
    for path in tqdm(paths):
        peptide_name = path.split("/")[1].split("_")[0]
        
        # get the distance matrix
        distance_matrix = get_pairdistances(path)
        
        # Count interchain contacts
        num_interchain_contacts = count_interchain_contacts(distance_matrix, path)
        
        # Update the dictionary with the count
        if peptide_name not in interchain_contacts:
            interchain_contacts[peptide_name] = {}
        
        interchain_contacts[peptide_name][f"{type}_intercontacts"] = num_interchain_contacts

        if interchain_contacts[peptide_name][f"{type}_intercontacts"] > 10:
            plot_flag = False
        if count < 10: plot_flag = True
        else: plot_flag = False
        if plot_flag:
            # plot the distance matrix
            plot_pairdistances(distance_matrix, path)
            # plot the contact map
            get_contactmap(distance_matrix, path)
        count += 1
# %%
# turn the dictionary into a dataframe where the index is the peptide name, and the columns are the number of interchain contacts for each type
df_interchain_contacts = pd.DataFrame.from_dict(interchain_contacts, orient='index')

# Reset index to make sure the peptide names are used as row labels
df_interchain_contacts.reset_index(inplace=True)
df_interchain_contacts.rename(columns={'index': 'peptide'}, inplace=True)

# make barplot of interchain contacts
df_interchain_contacts.plot(x='peptide', kind='bar', figsize=(15, 8))

# output the dataframe to a csv
df_interchain_contacts.to_csv('interchain_contacts.csv', index=False)