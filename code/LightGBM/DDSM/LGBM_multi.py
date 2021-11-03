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

def avgfeature(CCdf, MLOdf):
        avg = pd.DataFrame(CCdf.iloc[:,:1408].copy())
        for i in range(1408):
                avg[avg.columns[i]] = (CCdf[CCdf.columns[i]] + MLOdf[MLOdf.columns[i]])/2
        return avg

def getview (df):
    return df[df.index%2==0].reset_index(drop=True), df[df.index%2!=0].reset_index(drop=True)

df_train = pd.read_csv("/home/sam/Mammography/code/data/ddsm/featureddsmmultitrain.csv", header =None) #23536 rows, 1410 columns
df_test = pd.read_csv("/home/sam/Mammography/code/data/ddsm/featureddsmmultivalid.csv", header =None) #5064 rows
df_holdout = pd.read_csv("/home/sam/Mammography/code/data/ddsm/featureddsmmultiholdout.csv", header =None) #5084 rows
"""
df_train = pd.read_csv("/home/sam/Mammography/code/data/ddsm/featureddsmensembletrain.csv", header =None) #23536 rows, 1410 columns
df_test = pd.read_csv("/home/sam/Mammography/code/data/ddsm/featureddsmensemblevalid.csv", header =None) #5064 rows
df_holdout = pd.read_csv("/home/sam/Mammography/code/data/ddsm/featureddsmensembleholdout.csv", header =None) #5084 rows
"""
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

X_train = pd.concat((avgfeature(Ldf_trainCC,Ldf_trainMLO), avgfeature(Rdf_trainCC,Rdf_trainMLO)), axis = 0).to_numpy()
ybi_train = getmax(list(Ldf_trainCC.iloc[:,1408]), list(Ldf_trainMLO.iloc[:,1408])) + getmax(list(Rdf_trainCC.iloc[:,1408]), list(Rdf_trainMLO.iloc[:,1408]))
ybi_train = np.array(ybi_train)
yden_train = getmax(list(Ldf_trainCC.iloc[:,1409]), list(Ldf_trainMLO.iloc[:,1409])) + getmax(list(Rdf_trainCC.iloc[:,1409]), list(Rdf_trainMLO.iloc[:,1409]))

#%%
X_test = pd.concat((avgfeature(Ldf_testCC,Ldf_testMLO), avgfeature(Rdf_testCC,Rdf_testMLO)), axis = 0)
ybi_test = getmax(list(Ldf_testCC.iloc[:,1408]), list(Ldf_testMLO.iloc[:,1408])) + getmax(list(Rdf_testCC.iloc[:,1408]), list(Rdf_testMLO.iloc[:,1408]))
yden_test = getmax(list(Ldf_testCC.iloc[:,1409]), list(Ldf_testMLO.iloc[:,1409])) + getmax(list(Rdf_testCC.iloc[:,1409]), list(Rdf_testMLO.iloc[:,1409]))


LX_ho =  avgfeature(Ldf_hoCC,Ldf_hoMLO)
RX_ho = avgfeature(Rdf_hoCC,Rdf_hoMLO)

Lybi_ho = getmax(list(Ldf_hoCC.iloc[:,1408]), list(Ldf_hoMLO.iloc[:,1408]))
Rybi_ho = getmax(list(Rdf_hoCC.iloc[:,1408]), list(Rdf_hoMLO.iloc[:,1408]))
Lyden_ho = getmax(list(Ldf_hoCC.iloc[:,1409]), list(Ldf_hoMLO.iloc[:,1409]))
Ryden_ho = getmax(list(Rdf_hoCC.iloc[:,1409]), list(Rdf_hoMLO.iloc[:,1409]))

birad1_ind = np.where(ybi_train == 1)[0] #return tuple
birad2_ind = np.where(ybi_train == 2)[0]
birad3_ind = np.where(ybi_train == 3)[0]
birad4_ind = np.where(ybi_train == 4)[0]
birad5_ind = np.where(ybi_train == 5)[0]
initial_idx = np.concatenate((birad1_ind[:200], birad2_ind[:200], birad3_ind[:200], birad4_ind[:200], birad5_ind))

X_initial = X_train[initial_idx]
y_initial = ybi_train[initial_idx]

"""
# %%
# build the lightgbm model
clfbi = lgbm.LGBMClassifier(aapplication = "multiclass", learning_rate = 0.09, num_iterations= 900, num_leaves = 100,\
boosting_type="goss", max_depth = 3, max_bin = 35,class_weight={0:1, 1:5, 2:2},\
task ="train", objective = "multiclass", num_classes =3, seed = 100, data_random_seed = 20, bagging_seed = 20,\
is_unbalance = True, first_metric_only = True)

clfbi.fit(X_train, ybi_train, verbose=True, eval_set= (X_test, ybi_test), early_stopping_rounds =200, eval_metric = 'auc_mu')

#holdout
print("_________HOLDOUT_________")

print("__LEFT__")
print("BIRAD:")
Lybi_pred = clfbi.predict(LX_ho)
PrintResult( Lybi_pred, Lybi_ho )


print("__RIGHT__")
print("BIRAD:")
Rybi_pred = clfbi.predict(RX_ho)
PrintResult( Rybi_pred, Rybi_ho )

#SIDE
print("__SIDE__")
print("___BIRAD___")
y_pred_side = list(Lybi_pred) + list(Rybi_pred)
y_side = list(Lybi_ho) + list(Rybi_ho)
PrintResult(y_pred_side, y_side)

#STUDY
ybi = getmax(Lybi_ho, Rybi_ho)
ybi_pred = getmax(Lybi_pred, Rybi_pred)
print('-------STUDY-------')
print("____BIRAD___")
PrintResult( ybi_pred, ybi )
"""

clfden = lgbm.LGBMClassifier(application = "multiclass", num_iterations=700,num_leaves = 50,\
boosting_type="goss", max_depth = 5, max_bin=60,class_weight={1:7, 2:3, 3:4, 4:4},\
task = "train", num_classes = 4, seed = 200, data_random_seed = 200, bagging_seed = 200, first_metric_only = True)
clfden.fit(X_train, yden_train, verbose=True, eval_set = (X_test, yden_test),  eval_metric = 'auc_mu', early_stopping_rounds = 150)  #, eval_metric = 'auc_mu', early_stopping_rounds = 50

print("_________HOLDOUT_________")
print("\n __LEFT__")
print("*** DENSITY:")
Lyden_pred = clfden.predict(LX_ho)
PrintResult( Lyden_pred, Lyden_ho )

print("\n __RIGHT__")
print("DENSITY:")
Ryden_pred = clfden.predict(RX_ho)
PrintResult( Ryden_pred, Ryden_ho )

print("__SIDE__")
print("___DENSITY___")
y_pred_side = list(Lyden_pred) + list(Ryden_pred)
y_side = list(Lyden_ho) + list(Ryden_ho)
PrintResult(y_pred_side, y_side)

print("__STUDY__")
yden = getmax(Lyden_ho, Ryden_ho)
yden_pred = getmax(Lyden_pred, Ryden_pred)
print("____DENSITY___")
PrintResult( yden_pred, yden )


# save the model to disk
#filename = '/home/single4/mammo/mammo/sam/multiview/modelLGBM/LGBM_1024_Right.sav'
#pickle.dump(clfR, open(filename, 'wb'))
#filename = '/home/single4/mammo/mammo/sam/multiview/modelLGBM/Den_512_lgbmtop1.sav'
#pickle.dump(clfden, open(filename, 'wb'))


# %%
