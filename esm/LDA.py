#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Read the CSV of ThT / pFTAA amyloid classification
amyloid_df = pd.read_csv('true_labels_cutoff15.csv', index_col=0)['either']
amyloid_df = amyloid_df.rename('Classification')  # Rename the column to 'Classification'

# Read the embeddings
labels = ['3B', '15B']
embeddings_dfs = {label: pd.read_csv(f'{label}_embeddings.csv', index_col=0) for label in labels}

# Merge classification data with embeddings data
for label in labels:
    embeddings_dfs[label] = embeddings_dfs[label].merge(amyloid_df, left_index=True, right_index=True)

def plot_lda_histogram(embeddings_df, label):
    # Drop NaN values
    clean_embeddings_df = embeddings_df.dropna()

    # Perform LDA
    lda = LDA(n_components=1)
    lda_result = lda.fit_transform(clean_embeddings_df.iloc[:, :-1], clean_embeddings_df['Classification'])

    # Separate projections based on class
    amyloid_projection = lda_result[clean_embeddings_df['Classification'] == True]
    non_amyloid_projection = lda_result[clean_embeddings_df['Classification'] == False]

    # Plotting histograms
    plt.figure(figsize=(8, 6))
    plt.hist(amyloid_projection, bins=30, alpha=0.5, label='Amyloid', color='tab:orange')
    plt.hist(non_amyloid_projection, bins=30, alpha=0.5, label='Non-Amyloid', color='tab:blue')
    plt.title(f'LDA Projection Histogram for {label}', fontsize=15)
    plt.xlabel('Projection on LD', fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

for label, df in embeddings_dfs.items():
    plot_lda_histogram(df, label)
