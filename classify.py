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

for i, metric in enumerate(tqdm(metrics)):
    X = metric.reshape(-1,1)
    model = LogisticRegression()
    model.fit(X,y)

