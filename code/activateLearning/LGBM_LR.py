
#This file is used for TRAINING and TESTING in Activate Learning for studies
#%%
import pandas as pd 
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import lightgbm as lgbm

import pickle 
# ignore warnings
import warnings
warnings.filterwarnings("ignore")
#%%
import pandas as pd
def PrintResult(Nmodel, X, y ):
    # predict the results
    y_pred=Nmodel.predict(X)

    # view accuracy
    print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y, y_pred)))

    #confusion matrix
    cm = confusion_matrix(y, y_pred)
    print('Confusion matrix\n\n', cm)

    #classification metric
    print(classification_report(y, y_pred))
    print("F1 macro:", f1_score(y, y_pred, average = 'macro'))


def getdataframe(df):
    #function to get dataframe frome studies
    Left = df.drop(df.columns[1024:2048], axis = 1)
    Left.columns = range(Left.shape[1])
    Right = df.drop(df.columns[0:1024], axis = 1)
    Right.columns = range(Right.shape[1])
    temp = pd.concat([Left, Right], axis = 0, )
    return temp.reset_index().drop(columns=['index'])

def getmax(L, R):
    res = []
    for i in range(len(L)):
        res.append(max(L[i], R[i]))
    return res

def avgfeature(df):
        avg = pd.DataFrame(df.iloc[:,:512].copy())
        for i in range(512):
                avg[avg.columns[i]] = (df[df.columns[i]] + df[df.columns[i+512]])/2
        return avg
#%%
df_train = pd.read_csv("/home/sam/Mammography/code/data/beststudyfeature_multitrain.csv", header =None)
df_test = pd.read_csv("/home/sam/Mammography/code/data/beststudyfeature_multivalid.csv", header =None)
df_holdout = pd.read_csv("/home/sam/Mammography/code/data/beststudyfeature_multiholdout.csv", header =None)

df_train = getdataframe(df_train)
df_test = getdataframe(df_test)
df_holdout = getdataframe(df_holdout)

Xdf_train = avgfeature(df_train)
Xdf_test = avgfeature(df_test)
Xdf_holdout = avgfeature(df_holdout)

Ldf_train = Xdf_train.iloc[0:5884, :]
Rdf_train = Xdf_train.iloc[5884:, :]
Ldf_holdout = Xdf_holdout.iloc[0:1272,:]
Rdf_holdout = Xdf_holdout.iloc[1272:,:]
Ldf_test = Xdf_test.iloc[0:1267,:]
Rdf_test = Xdf_test.iloc[1267:,:]
#%%

X_train = Xdf_train
ybi_train = df_train.iloc[0:5884,1024].to_list() + df_train.iloc[5884:,1025].to_list() #concat label of Left Birad [0:5884,1024] và Right Birad [5884:,1025]
yden_train = df_train.iloc[0:5884,1026].to_list() + df_train.iloc[5884:,1027].to_list()

LX_test = Ldf_test
RX_test = Rdf_test
Lybi_test = df_test.iloc[0:1267,1024] 
Rybi_test = df_test.iloc[1267:,1025]
Lyden_test = df_test.iloc[0:1267, 1026].to_numpy()


LX_ho = Ldf_holdout
RX_ho = Rdf_holdout

Lybi_ho = df_holdout.iloc[0:1272,1024] 
Rybi_ho = df_holdout.iloc[1272:,1025] 
Lyden_ho = df_holdout.iloc[0:1272,1026].to_numpy()
Ryden_ho = df_holdout.iloc[1272:,1027].to_numpy()


# assemble initial data
"""
LX_train = LX_train.to_numpy()
Lybi_train = Lybi_train.to_numpy()
RX_train = RX_train.to_numpy()
Rybi_train = Rybi_train.to_numpy()
"""
X_train = X_train.to_numpy()
ybi_train = np.array(ybi_train)
yden_train = np.array(yden_train)

LX_test = LX_test.to_numpy()
RX_test = RX_test.to_numpy()
Lybi_test = Lybi_test.to_numpy()
Rybi_test = Rybi_test.to_numpy()

LX_hold = LX_ho.to_numpy()
Lybi_hold = Lybi_ho.to_numpy()
RX_hold = RX_ho.to_numpy()
Rybi_hold = Rybi_ho.to_numpy()
#%%
#n_initial = 100
#initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
#birad 1->5: 3091 1731 658 311 66
"""
birad1_ind = np.where(ybi_train == 0)[0] #return tuple
birad2_ind = np.where(ybi_train == 1)[0]
birad3_ind = np.where(ybi_train == 2)[0]
birad4_ind = np.where(ybi_train == 3)[0]
birad5_ind = np.where(ybi_train == 4)[0]
initial_idx = np.concatenate((birad1_ind[-250:], birad2_ind[-250:], birad3_ind[-250:], birad4_ind[-250:], birad5_ind))
"""
den0_ind = np.where(yden_train == 0)[0] #return tuple
den1_ind = np.where(yden_train == 1)[0]
den2_ind = np.where(yden_train == 2)[0]
den3_ind = np.where(yden_train == 3)[0]
print(len(den0_ind))
print(len(den1_ind))
print(len(den2_ind))
print(len(den3_ind))

initial_idx = np.concatenate((den0_ind, den1_ind[-300:], den2_ind[-300:], den3_ind[-300:]))

X_initial = X_train[initial_idx]
y_initial = yden_train[initial_idx]


#%%
# initializing the learner

cls = lgbm.LGBMClassifier(application = "multiclass", learning_rate = 0.1, num_iterations=45, num_leaves = 100,\
boosting_type="dart", max_depth = 5, max_bin = 200,\
task ="train", objective = "multiclass", num_classes =4, seed = 100, data_random_seed = 100, bagging_seed = 100, \
is_unbalance = True) 
   
cls.fit(X_initial, y_initial, verbose=True, eval_set= (LX_test, Lyden_test))
"""
Lcls = pickle.load(open('/home/sam/Mammography/code/modelLGBM/LGBM_512_avg_top_left.sav', "rb"))
Rcls = pickle.load(open('/home/sam/Mammography/code/modelLGBM/LGBM_512_avg_top_right.sav', "rb"))
"""

#holdout
print("_________HOLDOUT_________")
print("LEFT DENSITY:")
PrintResult(cls, LX_hold, Lyden_ho )
print("RIGHT DENSITY:")
PrintResult(cls, RX_hold, Ryden_ho )

#STUDY
#L_bi = cls.predict(LX_hold)
#R_bi = cls.predict(RX_hold)
L_den = cls.predict(LX_ho)
R_den = cls.predict(RX_ho)
#ybi = getmax(list(Lybi_ho), list(Rybi_ho))
yden = getmax(list(Lyden_ho), list(Ryden_ho))
#ybi_pred = getmax(L_bi, R_bi)
yden_pred = getmax(L_den, R_den)

print("____DENSITY___")
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(yden, yden_pred)))
cm = confusion_matrix(yden, yden_pred)
print('Confusion matrix\n\n', cm)

print(classification_report(yden, yden_pred))
print("F1 macro:", f1_score(yden, yden_pred, average = 'macro'))

filename = "/home/sam/Mammography/code/modelLGBM/LGBM_512_avg_DENSITY.sav"
pickle.dump(cls, open(filename, 'wb'))

