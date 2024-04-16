#%%
import pandas as pd
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
# from matplotlib.cm import get_cmap

# get the peptide names
tht_df = pd.read_csv('ThT_PAM4_paper.csv',index_col=0)
peptides = tht_df.columns[:-1] # ignore the control column

# Define the base folders to search within (these are the types of AF2 approaches)
types = ['multimer', 'multimer_5rec', '5U', '10U']

# Define the models to search for, including a 'best' model which is rank 1
models = ['best', 'model_1', 'model_2', 'model_3', 'model_4', 'model_5']

# Initialize a nested dictionary to hold the paths
files_dict = {type: {model: {} for model in models} for type in types}

for peptide in tqdm(peptides):
    for type in types:
        # Handle the 'best' case separately
        best_pattern = f'{type}/{peptide}_scores_rank_001*.json'
        best_files = glob(best_pattern)
        if best_files:
            files_dict[type]['best'][peptide] = best_files[0]
        else:
            print(f"No 'best' file found for {peptide} in {type}.")

        # Loop through each model number
        for model in models[1:]:  # Skip 'best' since it's already handled
            pattern = f'{type}/{peptide}_*_{model}_seed_000.json'
            files = glob(pattern)
            if files:
                files_dict[type][model][peptide] = files[0]
            else:
                print(f"No file found for {peptide} with {model} in {type} directory.")

#%%
# create dictionary where each metric lives (e.g. 'pLDDT_best', 'LIS_best')
my_metrics = ['plddt','LIS','LIA','pae_contacts','intercontacts','pTM','ipTM']
models_and_avg = ['best','model_1','model_2','model_3','model_4','model_5','avg']
metrics = {f'{metric}_{model}':{} for metric in my_metrics for model in models_and_avg}

# populate each metric with subdictionaries for type and peptide
for metric in metrics:
    metrics[metric] = {type: {peptide: None for peptide in peptides} for type in types}

# delete iptm for non-multimer models (ipTM is not defined for regular AF2)


# funtion to delete U's in the pLDDT or pAE scores
# scores can be either plddt (list) or pae (list of lists)
def delete_U_scores(scores, type, metric):
    if type == '5U': ranges_to_remove = [(15, 20), (35, 40), (55, 60), (75, 80)]
    elif type == '10U': ranges_to_remove = [(15, 25), (40, 50), (65, 75), (90, 100)]

    if metric == 'plddt':
        new_scores = [item for index, item in enumerate(scores) if not any(start <= index < end for start, end in ranges_to_remove)]
        assert len(new_scores) == 75, f"Expected length 75, got {len(new_scores)}"
    elif metric == 'pae':
        new_scores = np.array(scores)  # Convert to numpy array for easier slicing
        # Calculate the indices to keep (invert the logic to find indices to remove)
        indices_to_keep = set(range(len(scores)))  # Start with all indices
        for start, end in ranges_to_remove:
            indices_to_remove = set(range(start, end))
            indices_to_keep -= indices_to_remove  # Remove the indices in the specified ranges
        indices_to_keep = sorted(list(indices_to_keep))  # Sort and listify
        # Keep only the rows and columns with the indices in indices_to_keep
        new_scores = new_scores[np.ix_(indices_to_keep, indices_to_keep)]
        assert new_scores.shape == (75, 75), f"Expected shape (75,75), got {new_scores.shape}"
    return new_scores

def get_plddt(scores, type):
    all_plddts = scores['plddt']
    if type in ['5U','10U']:
        all_plddts = delete_U_scores(all_plddts, type, 'plddt')
    assert len(all_plddts) == 75, f'Expecting 75 plddts, but seeing {len(all_plddts)} plddts'
    plddt = np.mean(all_plddts)
    return plddt
    
# make mask of intra-chain contacts
n_residues_per_chain = 15; n_chains = 5
interchain_mask = np.ones((n_residues_per_chain * n_chains, n_residues_per_chain * n_chains), dtype=bool)
for i in range(n_chains):
    start_idx = i * n_residues_per_chain
    end_idx = start_idx + n_residues_per_chain
    interchain_mask[start_idx:end_idx, start_idx:end_idx] = False

def get_LIS_LIA(pae):
    pae_low_inter = pae < 12
    pae_low_inter[~interchain_mask] = False
    pae_subset = pae[pae_low_inter]
    pae_subset_rescaled = pae_subset / 12 * -1 + 1
    LIS = np.mean(pae_subset_rescaled)
    LIA = np.sum(pae_low_inter) / np.sum(interchain_mask)
    if LIS == np.nan: LIS = 0
    if LIA == np.nan: LIA = 0
    return LIS, LIA

def get_pae_intercontacts(pae,distance_path):
    # Note: this pae should be after deleting U scores
    contact_mask = pd.read_csv(distance_path,index_col=0).values < 0.8
    intercontact_mask = np.logical_and(contact_mask,interchain_mask)
    # count inter_contacts
    num_intercontacts = np.sum(intercontact_mask)
    # select just these elements of the pae matrix
    pae_contacts = np.mean(pae[intercontact_mask])
    return num_intercontacts, pae_contacts


#%%
# collect all data at once
for peptide in tqdm(peptides):
    for model in models:
        for type in types:
            # load scores
            file = files_dict[type][model][peptide]
            with open(file) as f:
                scores = json.load(f)

            # get plddts
            plddt = get_plddt(scores,type)
            metrics[f'plddt_{model}'][type][peptide] = plddt

            # get pTMs
            ptm = scores['ptm']
            metrics[f'pTM_{model}'][type][peptide] = ptm

            # get iPTMs for multimer models
            if type in ['multimer','multimer_5rec']:
                iptm = scores['iptm']
                metrics[f'ipTM_{model}'][type][peptide] = iptm

            # get pae matrix
            pae = np.array(scores['pae'])
            if type in ['5U','10U']:
                pae = delete_U_scores(pae,type,'pae')

            # get LIS scores and LIA scores
            LIS, LIA = get_LIS_LIA(pae) # LIA is actually a fraction of all possible intercontacts
            metrics[f'LIS_{model}'][type][peptide] = LIS
            metrics[f'LIA_{model}'][type][peptide] = LIA

            # get pae_contacts (pae values for interchain contact regions)
            distance_path = f'distance_matrices/{type}/{peptide}_{model}.csv'
            num_intercontacts, pae_contacts = get_pae_intercontacts(pae,distance_path)
            metrics[f'intercontacts_{model}'][type][peptide] = num_intercontacts
            metrics[f'pae_contacts_{model}'][type][peptide] = pae_contacts

#%% fill out avg metrics in the metrics dictionary
# i.e. the average of the 5 AF2 models
for peptide in tqdm(peptides):
    for metric in my_metrics:
        for type in types:
            if np.logical_and(metric == 'ipTM', type in ['5U','10U']):
                continue
            else:
                mean_models = np.sum([metrics[f'{metric}_{model}'][type][peptide] for model in models[1:]]) / 5
                metrics[f'{metric}_avg'][type][peptide] = mean_models
models.extend(['avg'])
#%%
features = []
columns = []
for metric in my_metrics:
    for model in models:
        for type in types:
            if np.logical_and(metric=='ipTM',type in ['5U','10U']):
                continue
            columns.append(f'{metric}_{model}_{type}')
            metrics_list = []
            for peptide in peptides:
                metrics_list.append(metrics[f'{metric}_{model}'][type][peptide])
            features.append(metrics_list)
# put into df
features = np.array(features).T
df_features = pd.DataFrame(features,columns=columns,index=peptides)
# save csv
df_features.to_csv('features.csv')

# %%
