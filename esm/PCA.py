#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# read the csv of ThT / pFTAA amyloid classification
amyloid_df = pd.read_csv('true_labels_cutoff15.csv',index_col=0)['either']
# rename the column to 'Classification'
amyloid_df = amyloid_df.rename('Classification')

#%%

# read the embeddings
labels = ['3B','15B']
embeddings_dfs = {label: pd.read_csv(f'{label}_embeddings.csv', index_col=0) for label in labels}

# Merge classification data with embeddings data
for label in labels:
    # merge the embeddings with the amyloid classification
    embeddings_dfs[label] = embeddings_dfs[label].merge(amyloid_df, left_index=True, right_index=True)
#%%

def plot_pca_with_colors(embeddings_df, label):
    clean_embeddings_df = embeddings_df.dropna()

    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(clean_embeddings_df.iloc[:, :-2])
    
    # Define colors based on 'Classification'
    colors = clean_embeddings_df['Classification'].map({False: 'tab:blue', True: 'tab:orange'})
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5, c=colors)
    plt.title(f'PCA Plot for {label}', fontsize=15)
    plt.xlabel('Principal Component 1', fontsize=15)
    plt.ylabel('Principal Component 2', fontsize=15)
    plt.xticks(fontsize=15); plt.yticks(fontsize=15)
    # Create a legend
    classes = ['amyloid', 'non-amyloid']
    class_colors = ['tab:blue', 'tab:orange']
    recs = []
    for i in range(0,len(class_colors)):
        recs.append(plt.Rectangle((0,0),1,1,fc=class_colors[i]))
        plt.legend(recs, classes, loc='best', fontsize=15)
    plt.show()


def plot_pca(embeddings_df, label):
    # Replace NaN values with the mean of each column (if any NaNs exist)
    # embeddings_df = embeddings_df.fillna(embeddings_df.mean())

    # Perform PCA
    n_components = 5
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(embeddings_df.iloc[:, :-1])

    # plot the explained variance ratio
    plt.figure(figsize=(8, 6))
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_ * 100)
    plt.plot(np.arange(1, n_components + 1), cumulative_variance)
    plt.xticks(np.arange(n_components)+1, fontsize=15)
    plt.xticks(fontsize=15); plt.yticks(fontsize=15)
    plt.xlabel('Number of Components', fontsize=15)
    plt.ylabel('Explained Variance (%)', fontsize=15)
    plt.show()
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)
    plt.title(f'esm_{label} model', fontsize=15)
    plt.xlabel('Principal Component 1', fontsize=15)
    plt.ylabel('Principal Component 2', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)
    plt.show()

for label, df in embeddings_dfs.items():
    # plot_pca(df, label)

    plot_pca_with_colors(df, label)

# %%
