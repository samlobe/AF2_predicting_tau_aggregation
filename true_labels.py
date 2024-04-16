#%%
import pandas as pd
import matplotlib.pyplot as plt

plot_flag = False
output_flag = True

# calling it aggregating if it is x-fold more signal than 
cutoff = 1.5 

# read the ThT data
df_ThT = pd.read_csv('ThT_PAM4_paper.csv',index_col=0)

# add a row that is the mean of the ThT data
df_ThT.loc['mean'] = df_ThT.mean()

# divide all the data by the mean of the control data
df_ThT_norm = df_ThT / df_ThT['Control']['mean']

if plot_flag == True:
    # plot a bar plot of the ThT data
    df_ThT_norm.loc['mean'].plot(kind='bar')

    # set y-max to 1.5
    plt.ylim([0,1.5])
    # plot horizontal line at y=1
    plt.axhline(y=1, color='black', linestyle='--')

    # print the columns that have a normalized mean more than cutoff (e.g. 1.5)
    print(df_ThT_norm.columns[df_ThT_norm.loc['mean'] > cutoff])
    # color these columns red
    df_ThT_norm.loc['mean'].plot(kind='bar',color=['red' if x > cutoff else 'blue' for x in df_ThT_norm.loc['mean']])

#%%
# read the pFTAA data
df_pFTAA = pd.read_csv('pFTAA_PAM4_paper.csv',index_col=0)

# add a row that is the mean of the pFTAA data
df_pFTAA.loc['mean'] = df_pFTAA.mean()

# divide all the data by the mean of the control data
df_pFTAA_norm = df_pFTAA / df_pFTAA['Control']['mean']

if plot_flag == True:
    # plot a bar plot of the pFTAA data
    df_pFTAA_norm.loc['mean'].plot(kind='bar')

    # set y-max to 1.5
    plt.ylim([0,1.5])
    # plot horizontal line at y=1
    plt.axhline(y=1, color='black', linestyle='--')

    # print the columns that have a normalized mean more than 1.5
    print(df_pFTAA_norm.columns[df_pFTAA_norm.loc['mean'] > cutoff])

    # color these columns red
    df_pFTAA_norm.loc['mean'].plot(kind='bar',color=['red' if x > cutoff else 'blue' for x in df_pFTAA_norm.loc['mean']])

# %%
# print how many have a normalized mean more than 1.5 for ThT and pFTAA
print('ThT:',sum(df_ThT_norm.loc['mean'] > cutoff))
print('pFTAA:',sum(df_pFTAA_norm.loc['mean'] > cutoff))

# find the set that have a normalized mean more than 1.5 for either ThT or pFTAA
true_agg = set(df_ThT_norm.columns[df_ThT_norm.loc['mean'] > cutoff]) | set(df_pFTAA_norm.columns[df_pFTAA_norm.loc['mean'] > 1.5])
print('ThT or pFTAA:',len(true_agg))
print(true_agg)

#%%
# create a csv with the true labels for the ThT data, pFTAA data, and the either set
df_true_labels = pd.DataFrame(index=df_ThT_norm.columns,columns=['ThT','pFTAA','either'])
df_true_labels['ThT'] = df_ThT_norm.loc['mean'] > cutoff
df_true_labels['pFTAA'] = df_pFTAA_norm.loc['mean'] > cutoff
df_true_labels['either'] = df_true_labels['ThT'] | df_true_labels['pFTAA']
# remove the control column
df_true_labels = df_true_labels.drop('Control')
df_true_labels.to_csv('true_labels.csv')

# # sum the number of true labels for ThT, pFTAA, and the either set
# print('ThT:',sum(df_true_labels['ThT']))
# print('pFTAA:',sum(df_true_labels['pFTAA']))
# print('either:',sum(df_true_labels['either']))