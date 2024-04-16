#%%
import numpy as np
import pandas as pd
from glob import glob
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
        # print(distance_matrix)
        # print(np.shape(distance_matrix))
    assert distance_matrix.shape == (75, 75), f'Expected 75x75 array and got {np.size(distance_matrix)}'

    # convert to pandas df
    chains = ['A', 'B', 'C', 'D', 'E']
    resids = np.arange(1,16)
    indices = [f'{chain}{resid}' for chain in chains for resid in resids]
    distance_df = pd.DataFrame(distance_matrix, columns=indices, index=indices)
    return distance_df

def get_chain_info(peptide_path):
    AF2_pdb = md.load(peptide_path)
    topology = AF2_pdb.topology
    
    if peptide_path.split("/")[0] in ['multimer','multimer_5rec']:
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
    if not os.path.exists(f'distance_matrices/{type}'):
        os.makedirs(f'distance_matrices/{type}')

# interchain contacts dict
# interchain_contacts = {type: {model: {} for model in models} for type in types}

#%%
# loop through all the peptides and categories and store their distance array in csv
for peptide in tqdm(peptides):
    for type in types:
        for model in models:
            path = files_dict[type][model][peptide]
            # get pairdist array
            distance_df = get_pairdistances(path)
            # output as pandas dataframe in distance_matrices directory
            distance_df.to_csv(f'distance_matrices/{type}/{peptide}_{model}.csv')
