#%% 
import pandas as pd
import matplotlib.pyplot as plt

# calling it aggregating if it is x-fold more signal than 
cutoff = 1.5 
df_ThT = pd.read_csv('ThT_PAM4_paper.csv',index_col=0)
df_ThT.loc['mean'] = df_ThT.mean()
df_ThT_norm = df_ThT / df_ThT['Control']['mean']
df_pFTAA = pd.read_csv('pFTAA_PAM4_paper.csv',index_col=0)
df_pFTAA.loc['mean'] = df_pFTAA.mean()
df_pFTAA_norm = df_pFTAA / df_pFTAA['Control']['mean']

# create a csv with the true labels for the ThT data, pFTAA data, and the either set
df_true_labels = pd.DataFrame(index=df_ThT_norm.columns,columns=['ThT','pFTAA','either'])
df_true_labels['ThT'] = df_ThT_norm.loc['mean'] > cutoff
df_true_labels['pFTAA'] = df_pFTAA_norm.loc['mean'] > cutoff
df_true_labels['either'] = df_true_labels['ThT'] | df_true_labels['pFTAA']
# remove the control column
df_true_labels = df_true_labels.drop('Control')
df_true_labels.to_csv(f'true_labels_cutoff{int(cutoff*10)}.csv')



#%%
features = pd.read_csv('features.csv',index_col=0)
labels = pd.read_csv('true_labels.csv',index_col=0)

# y = labels['either'].values
y = labels['ThT'].values


# interesting features:
# 'intercontacts_avg_10U' pretty good
# 'LIS_best_10U' & 'pae_contacts_best_10U' maybe good for strong aggregators? check this
# 'pae_contacts_model_5_10U

# bad: plddt, LIS_multimer, LIA_multimer

my_metrics = ['plddt','LIS','LIA','pae_contacts','intercontacts','pTM','ipTM']

feature = 'pae_contacts_best_10U'
x = features[feature].values
plt.hist(x[y],label='aggregation',bins=20,alpha=0.5)
plt.hist(x[~y],label='NO aggregation',bins=20,alpha=0.5)
plt.xlabel(feature,fontsize=15)
plt.ylabel('counts',fontsize=15)
plt.legend(fontsize=15)
plt.show()

# select 
#
# average the intercontacts of 
feature1 = 'intercontacts_best_10U'
feature2 = 'pTM_avg_10U'
# feature2 = 'LIA_avg_10U'
x1 = features[feature1]; x2 = features[feature2]
plt.scatter(x1[y],x2[y],label='aggregation',alpha=0.5)
plt.scatter(x1[~y],x2[~y],label='NO aggregation',alpha=0.5)
plt.xlabel(feature1,fontsize=15)
plt.ylabel(feature2,fontsize=15)
plt.legend(fontsize=12)

# write peptide in text below each blue point
annotate_it = features.index[y]
for i, annotation in enumerate(annotate_it):
    plt.annotate(annotation,(x1[y][i],x2[y][i]))