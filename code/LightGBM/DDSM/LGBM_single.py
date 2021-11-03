#%%
from os import device_encoding
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import lightgbm as lgbm
import pickle 
from sklearn.model_selection import train_test_split
# ignore warnings
import warnings
warnings.filterwarnings("ignore")

def PrintResult(y_pred,y ):
    # view accuracy
    print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y, y_pred)))

    #confusion matrix
    cm = confusion_matrix(y, y_pred)
    print('Confusion matrix\n\n', cm)

    #classification metric
    print(classification_report(y, y_pred))
    print("F1 macro:", f1_score(y, y_pred, average = 'macro'))



df_train = pd.read_csv("/home/sam/Mammography/code/data/ddsm/ddsmsingletrain.csv", header =None) #6108 rows, 1409 columns
df_holdout = pd.read_csv("/home/sam/Mammography/code/data/ddsm/ddsmsingleval.csv", header =None) #1528 rows, 1409 columns
df_train = df_train.astype({1408: int})
df_holdout = df_holdout.astype({1408:int})
X = df_train.iloc[:,:1408]
y = df_train.iloc[:,1408]
X_train = X
y_train = y
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_holdout, y_holdout = df_holdout.iloc[:,:1408], df_holdout.iloc[:,1408]

# %%
# build the lightgbm model
clf = lgbm.LGBMClassifier(num_iterations = 500, first_metric_only = True, max_depth = 2, learning_rate=0.14,seed = 300, data_random_seed = 200,\
    boosting_type="gbdt", num_leaves = 10) # bagging_freq = 5, bagging_fraction = 0.75

clf.fit(X_train, y_train, verbose=True, eval_set= (X_holdout, y_holdout), early_stopping_rounds =200, eval_metric = 'auc')

filename = '/home/sam/Mammography/code/modelLGBM/ddsm/single.sav'
pickle.dump(clf, open(filename, 'wb'))

print("Best interations: ", clf._best_iteration)
print("Best score: ", clf._best_score)
print("_________HOLDOUT_________")
print("CLASSIFICATION:")

y_pred = clf.predict(X_holdout)
PrintResult( y_pred, y_holdout )

