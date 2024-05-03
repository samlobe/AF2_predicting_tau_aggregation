#%%
import pandas as pd
import torch
import os 
import numpy as np

fragments = pd.read_csv('true_labels_cutoff15.csv',index_col=0).index

#%%
folders = ['esm2_3B_output','esm2_15B_output']
labels = ['3B','15B']

for i, folder in enumerate(folders):
    embeddings_list = []
    if labels[i] in ['3B','3B_unk']:
        layers = 36; n_embeddings = 2560
    else:
        layers = 48; n_embeddings = 5120

    for peptide in fragments:
        # Load the embeddings
        file = f'{folder}/{peptide}.pt'
        if os.path.exists(file):
            embeddings = torch.load(file)['mean_representations'][layers].detach().numpy()
            embeddings_list.append(embeddings)
        else: 
            print(f'File {file} does not exist')
            # output a row of NaNs
            embeddings_list.append(np.nan*np.ones(n_embeddings))

    # output csv of embeddings
    embeddings_df = pd.DataFrame(embeddings_list, index=fragments)
    # prepend 'embedding' to each column
    embeddings_df.columns = [f'embedding_{i}' for i in range(len(embeddings_df.columns))]

    embeddings_df.to_csv(f'{labels[i]}_embeddings.csv')
