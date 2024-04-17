#%%
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm

features = pd.read_csv('features.csv',index_col=0)
labels = pd.read_csv('true_labels_cutoff15.csv',index_col=0)

y = labels['either'].values # ThT or pFTAA aggregation
# y = labels['ThT'].values
# y = y.reshape(-1,1)

metrics = features.values.T
# replace nans with 0 (for LIS and pae_contacts metrics))
metrics[np.isnan(metrics)] = 0

for i, metric in enumerate(tqdm(metrics)):
    X = metric.reshape(-1,1)
    model = LogisticRegression()
    model.fit(X,y)
    probs = model.predict_proba(X)[:,1]
    fpr, tpr, thresholds = roc_curve(y,probs)
    roc_auc = auc(fpr,tpr)

    plt.plot(fpr,tpr,label=f'{features.columns[i]} (AUC = {roc_auc:.2f})')

plt.plot([0,1],[0,1],'k--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate',fontsize=15)
plt.ylabel('True Positive Rate',fontsize=15)
# plt.legend()
plt.show()

#%%
# make a dataframe with the AUC values
aucs = []
for i, metric in enumerate(tqdm(metrics)):
    X = metric.reshape(-1,1)
    model = LogisticRegression()
    model.fit(X,y)
    probs = model.predict_proba(X)[:,1]
    fpr, tpr, thresholds = roc_curve(y,probs)
    roc_auc = auc(fpr,tpr)
    aucs.append(roc_auc)

df_aucs = pd.DataFrame(aucs,index=features.columns,columns=['AUC'])

#%%
# sort the dataframe by AUC
df_aucs = df_aucs.sort_values(by='AUC',ascending=False)
# print the top 10
print(df_aucs.head(10))

# get custom metrics to plot the ROC
my_features = ['LIS_model_5_10U','intercontacts_best_10U','LIS_model_5_multimer','intercontacts_best_multimer','ipTM_best_multimer']

for feature in my_features:
    X = features[feature].values.reshape(-1,1)
    model = LogisticRegression()
    model.fit(X,y)
    probs = model.predict_proba(X)[:,1]
    fpr, tpr, thresholds = roc_curve(y,probs)
    roc_auc = auc(fpr,tpr)

    plt.plot(fpr,tpr,label=f'{feature} (AUC = {roc_auc:.2f})')

plt.legend()
plt.plot([0,1],[0,1],'k--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate',fontsize=15)
plt.ylabel('True Positive Rate',fontsize=15)
plt.show()

#%%
my_features = ['LIS_model_5_10U','LIS_model_5_5U','LIS_model_5_multimer','LIS_model_5_multimer_5rec']

for feature in my_features:
    X = features[feature].values.reshape(-1,1)
    model = LogisticRegression()
    model.fit(X,y)
    probs = model.predict_proba(X)[:,1]
    fpr, tpr, thresholds = roc_curve(y,probs)
    roc_auc = auc(fpr,tpr)

    plt.plot(fpr,tpr,label=f'{feature} (AUC = {roc_auc:.2f})')

plt.legend()
plt.plot([0,1],[0,1],'k--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate',fontsize=15)
plt.ylabel('True Positive Rate',fontsize=15)
plt.show()

#%%
# compare all LIS 10U models
my_features = ['LIS_model_1_10U','LIS_model_2_10U','LIS_model_3_10U','LIS_model_4_10U','LIS_model_5_10U','LIS_best_10U','LIS_avg_10U']

for feature in my_features:
    X = features[feature].values.reshape(-1,1)
    model = LogisticRegression()
    model.fit(X,y)
    probs = model.predict_proba(X)[:,1]
    fpr, tpr, thresholds = roc_curve(y,probs)
    roc_auc = auc(fpr,tpr)

    plt.plot(fpr,tpr,label=f'{feature} (AUC = {roc_auc:.2f})')

plt.legend()
plt.plot([0,1],[0,1],'k--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate',fontsize=15)
plt.ylabel('True Positive Rate',fontsize=15)
plt.show()

#%%
# compare all intercontacts_best
my_features = ['intercontacts_best_10U','intercontacts_best_5U','intercontacts_best_multimer','intercontacts_best_multimer_5rec']

for feature in my_features:
    X = features[feature].values.reshape(-1,1)
    model = LogisticRegression()
    model.fit(X,y)
    probs = model.predict_proba(X)[:,1]
    fpr, tpr, thresholds = roc_curve(y,probs)
    roc_auc = auc(fpr,tpr)

    plt.plot(fpr,tpr,label=f'{feature} (AUC = {roc_auc:.2f})')

plt.legend()
plt.plot([0,1],[0,1],'k--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate',fontsize=15)
plt.ylabel('True Positive Rate',fontsize=15)
plt.show()

#%%
# compare intercontacts between models
my_features = ['intercontacts_model_1_10U','intercontacts_model_2_10U','intercontacts_model_3_10U','intercontacts_model_4_10U','intercontacts_model_5_10U',
               'intercontacts_best_10U','intercontacts_avg_10U']

for feature in my_features:
    X = features[feature].values.reshape(-1,1)
    model = LogisticRegression()
    model.fit(X,y)
    probs = model.predict_proba(X)[:,1]
    fpr, tpr, thresholds = roc_curve(y,probs)
    roc_auc = auc(fpr,tpr)

    plt.plot(fpr,tpr,label=f'{feature} (AUC = {roc_auc:.2f})')

plt.legend()
plt.plot([0,1],[0,1],'k--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate',fontsize=15)
plt.ylabel('True Positive Rate',fontsize=15)
plt.show()

#%%
# print out the AUC for all the metrics beginning with "LIS"
print(df_aucs[df_aucs.index.str.startswith('LIS')])

#%%
# print out the AUC for all the metrics beginning with "intercontacts"
print(df_aucs[df_aucs.index.str.startswith('intercontacts')])

#%%
# plot a histogram of aggregators with LIS_model_5_10U and then nonaggregators
feature = 'LIS_model_5_10U'
feature = 'intercontacts_best_10U'
x = features[feature].values
plt.hist(x[y],label='aggregation',bins=20,alpha=0.5)
plt.hist(x[~y],label='NO aggregation',bins=20,alpha=0.5)
plt.xlabel(feature,fontsize=15)
plt.ylabel('counts',fontsize=15)
plt.legend(fontsize=15)

# #%%
# # Find the index for 'LIS_model_5_10U'
# base_feature_name = 'pae_contacts_avg_5U'
# base_index = features.columns.get_loc(base_feature_name)

# # Store the base feature data
# base_feature = metrics[base_index].reshape(-1, 1)

# # Prepare a list to store AUC values and feature names
# combined_auc = []

# # Iterate over all features to test them in combination with 'LIS_model_5_10U'
# for i, metric in enumerate(tqdm(metrics)):
#     if i != base_index:  # Skip the base feature itself
#         # Combine the base feature with the current feature
#         combined_feature = np.hstack((base_feature, metric.reshape(-1, 1)))
        
#         # Fit the logistic regression model
#         model = LogisticRegression()
#         model.fit(combined_feature, y)
        
#         # Calculate probabilities and AUC
#         probs = model.predict_proba(combined_feature)[:, 1]
#         fpr, tpr, thresholds = roc_curve(y, probs)
#         roc_auc = auc(fpr, tpr)
        
#         # Append the result to the list
#         combined_auc.append((features.columns[i], roc_auc))

# # Sort and print the best combinations based on AUC
# combined_auc.sort(key=lambda x: x[1], reverse=True)
# print("Top 5 feature combinations with their AUCs:")
# for feature, auc in combined_auc[:30]:
#     print(f"{base_feature_name} + {feature}: AUC = {auc:.4f}")
# # %%
# # plot the aggregators along intercontacts_best_10U + intercontacts_model_3_multimer_5rec
# feature1 = 'LIS_model_5_10U'
# feature2 = 'intercontacts_best_10U'
# x1 = features[feature1]; x2 = features[feature2]
# plt.scatter(x1[y],x2[y],label='aggregation',alpha=0.5)
# plt.scatter(x1[~y],x2[~y],label='NO aggregation',alpha=0.5)
# plt.xlabel(feature1,fontsize=15)
# plt.ylabel(feature2,fontsize=15)

# # write peptide in text below each blue point
# annotate_it = features.index[y]
# for i, annotation in enumerate(annotate_it):
#     plt.text(x1[y][i],x2[y][i],annotation,fontsize=10)

# plt.legend()
