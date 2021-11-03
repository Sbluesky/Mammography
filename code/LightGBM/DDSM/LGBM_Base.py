#%%
from os import device_encoding
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import lightgbm as lgbm
import pickle 
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

def getmax(L, R):
    res = []
    for i in range(len(L)):
        res.append(max(L[i], R[i]))
    return res

def getDirec (df):
    return df[df.index %4 <2].reset_index(drop=True), df[df.index %4 >=2].reset_index(drop=True) #left, right

def getview (df):
    return df[df.index%2==0].reset_index(drop=True), df[df.index%2!=0].reset_index(drop=True)

df_train = pd.read_csv("/home/sam/Mammography/code/data/ddsm/featureddsmensembletrain.csv", header =None) #23536 rows, 1410 columns
df_test = pd.read_csv("/home/sam/Mammography/code/data/ddsm/featureddsmensemblevalid.csv", header =None) #5064 rows
df_holdout = pd.read_csv("/home/sam/Mammography/code/data/ddsm/featureddsmensembleholdout.csv", header =None) #5084 rows

#LCC LMLO RCC RMLO
#%%
Ldf_train, Rdf_train = getDirec(df_train) #11768 rows    
Ldf_test, Rdf_test = getDirec(df_test)
Ldf_holdout, Rdf_holdout = getDirec(df_holdout)

Ldf_trainCC, Ldf_trainMLO = getview(Ldf_train) #5884 rows, 514 columns
Rdf_trainCC, Rdf_trainMLO = getview(Rdf_train)

Ldf_testCC, Ldf_testMLO = getview(Ldf_test)
Rdf_testCC, Rdf_testMLO = getview(Rdf_test)

Ldf_hoCC, Ldf_hoMLO = getview(Ldf_holdout)
Rdf_hoCC, Rdf_hoMLO = getview(Rdf_holdout)
# %%

X_train = df_train.iloc[:,0:1408].to_numpy()
ybi_train = df_train.iloc[:, 1408].to_numpy()
yden_train = df_train.iloc[:, 1409].to_numpy()

X_test = df_test.iloc[:,0:1408].to_numpy()
ybi_test = df_test.iloc[:, 1408].to_numpy()
yden_test = df_test.iloc[:, 1409].to_numpy()

###SINGLE###
LX_ho_CC,LX_ho_MLO = Ldf_hoCC.iloc[:,0:1408], Ldf_hoMLO.iloc[:,0:1408]
Lybi_ho_CC, Lybi_ho_MLO = Ldf_hoCC.iloc[:,1408], Ldf_hoMLO.iloc[:,1408]
Lyden_ho_CC,Lyden_ho_MLO = Ldf_hoCC.iloc[:,1409], Ldf_hoMLO.iloc[:,1409]

RX_ho_CC,RX_ho_MLO = Rdf_hoCC.iloc[:,0:1408], Rdf_hoMLO.iloc[:,0:1408]
Rybi_ho_CC, Rybi_ho_MLO = Rdf_hoCC.iloc[:,1408], Rdf_hoMLO.iloc[:,1408]
Ryden_ho_CC,Ryden_ho_MLO = Rdf_hoCC.iloc[:,1409], Rdf_hoMLO.iloc[:,1409]

"""
# %%
# build the lightgbm model
clfbi = lgbm.LGBMClassifier(application = "multiclass", learning_rate = 0.1, num_iterations=800, num_leaves = 100,\
boosting_type="goss", max_depth = 3, max_bin = 200,\
task ="train", objective = "multiclass", num_classes =3, seed = 10, data_random_seed = 100, bagging_seed = 100, \
is_unbalance = True, first_metric_only = True)

clfbi.fit(X_train, ybi_train, verbose=True, eval_set= (X_test, ybi_test), early_stopping_rounds =100, eval_metric = 'auc_mu')

#holdout

print("_________HOLDOUT_________")
print("__LEFT__")

print("BIRAD:")
y_pred_CC = clfbi.predict(LX_ho_CC)
y_pred_MLO = clfbi.predict(LX_ho_MLO)
Lybi_pred = getmax(y_pred_CC,y_pred_MLO)
Lybi = getmax(list(Lybi_ho_CC),list(Lybi_ho_MLO))
PrintResult( Lybi_pred, Lybi )

print("__RIGHT__")

print("BIRAD:")
y_pred_CC = clfbi.predict(RX_ho_CC)
y_pred_MLO = clfbi.predict(RX_ho_MLO)
Rybi_pred = getmax(y_pred_CC,y_pred_MLO)
Rybi = getmax(list(Rybi_ho_CC),list(Rybi_ho_MLO))
PrintResult( Rybi_pred, Rybi )

#SIDE
print("__SIDE__")
y_pred_side = list(Lybi_pred) + list(Rybi_pred)
y_side = list(Lybi) + list(Rybi)
PrintResult(y_pred_side, y_side)

#STUDY
#%%
ybi = getmax(list(Lybi), list(Rybi)) #SINGLE
ybi_pred = getmax(Lybi_pred, Rybi_pred)
print('-------STUDY-------')
print("____BIRAD___")
PrintResult( ybi_pred, ybi )
"""

clfden = lgbm.LGBMClassifier(application = "multiclass", num_iterations=300,\
boosting_type="goss", max_depth = 5, max_bin=200, class_weight={1:10, 2:3, 3:1, 4:1},\
task = "train", num_classes = 4, seed = 10, first_metric_only = True )
clfden.fit(X_train, yden_train, verbose=True, eval_set = (X_test, yden_test),  eval_metric = 'auc_mu', early_stopping_rounds = 50)  #, eval_metric = 'auc_mu', early_stopping_rounds = 50

print("__LEFT__")
print("*** \n DENSITY:")
y_pred_CC = clfden.predict(LX_ho_CC)
y_pred_MLO = clfden.predict(LX_ho_MLO)
Lyden_pred = getmax(y_pred_CC,y_pred_MLO)
Lyden = getmax(list(Lyden_ho_CC),list(Lyden_ho_MLO))
PrintResult( Lyden_pred, Lyden )

print("__RIGHT__")
print(" \n DENSITY:")
y_pred_CC = clfden.predict(RX_ho_CC)
y_pred_MLO = clfden.predict(RX_ho_MLO)
Ryden_pred = getmax(y_pred_CC,y_pred_MLO)
Ryden = getmax(list(Ryden_ho_CC),list(Ryden_ho_MLO))
PrintResult( Ryden_pred, Ryden )

print("__SIDE__")
y_pred_side = list(Lyden_pred) + list(Ryden_pred)
y_side = list(Lyden) + list(Ryden)
PrintResult(y_pred_side, y_side)

print("__STUDY__")
print("____DENSITY___")
yden = getmax(list(Lyden), list(Ryden)) #SINGLE
yden_pred = getmax(Lyden_pred, Ryden_pred)
PrintResult( yden_pred, yden )


# save the model to disk
#filename = '/home/single4/mammo/mammo/sam/multiview/modelLGBM/LGBM_1024_Right.sav'
#pickle.dump(clfR, open(filename, 'wb'))
#filename = '/home/single4/mammo/mammo/sam/multiview/modelLGBM/Den_512_lgbmtop1.sav'
#pickle.dump(clfden, open(filename, 'wb'))


# %%
