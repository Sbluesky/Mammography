#%%
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner, Committee
import pandas as pd

df_train = pd.read_csv("/home/sam/Mammography/code/studyfeature_multitrain.csv", header =None)
df_test = pd.read_csv("/home/sam/Mammography/code/studyfeature_multivalid.csv", header =None)
df_holdout = pd.read_csv("/home/sam/Mammography/code/studyfeature_multiholdout.csv", header =None)

X_train = df_train.iloc[:, 0:2048] #get 4 features
ybi_train = df_train.iloc[:,2052] + 1
yden_train = df_train.iloc[:,2053]

X_test = df_test.iloc[:,0:2048]
ybi_test = df_test.iloc[:,2052] + 1
yden_test = df_test.iloc[:,2053]

X_hold = df_holdout.iloc[:,0:2048]
ybi_hold = df_holdout.iloc[:,2052] + 1
yden_hold = df_holdout.iloc[:,2053]

# assemble initial data
X_train = X_train.to_numpy()
ybi_train = ybi_train.to_numpy()
X_hold = X_hold.to_numpy()
ybi_hold = ybi_hold.to_numpy()
# visualizing the classes
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(7, 7))
    pca = PCA(n_components=2).fit_transform(X_hold)
    plt.scatter(x=pca[:, 0], y=pca[:, 1], c=ybi_hold, cmap='viridis', s=50)
    plt.title('The Mammography dataset')
    plt.savefig('/home/sam/Mammography/report/HoldingOutDataDistribution.png')
    plt.show()
    
# %%
