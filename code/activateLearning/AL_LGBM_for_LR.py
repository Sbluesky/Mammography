
#This file is used for TRAINING and TESTING in Activate Learning for studies
#%%
import pandas as pd 
import numpy as np
from modAL.models import ActiveLearner
from modAL.multilabel import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import lightgbm as lgb; 
from modAL.uncertainty import uncertainty_sampling, entropy_sampling,margin_sampling
from modAL.batch import uncertainty_batch_sampling
from modAL.disagreement import max_std_sampling
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


LX_ho = Ldf_holdout
RX_ho = Rdf_holdout

Lybi_ho = df_holdout.iloc[0:1272,1024] 
Rybi_ho = df_holdout.iloc[1272:,1025] 
Lyden_ho = df_holdout.iloc[0:1272,1026]
Ryden_ho = df_holdout.iloc[1272:,1027]


# assemble initial data
"""
LX_train = LX_train.to_numpy()
Lybi_train = Lybi_train.to_numpy()
RX_train = RX_train.to_numpy()
Rybi_train = Rybi_train.to_numpy()
"""
X_train = X_train.to_numpy()
ybi_train = np.array(ybi_train)

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

birad1_ind = np.where(ybi_train == 0)[0] #return tuple
birad2_ind = np.where(ybi_train == 1)[0]
birad3_ind = np.where(ybi_train == 2)[0]
birad4_ind = np.where(ybi_train == 3)[0]
birad5_ind = np.where(ybi_train == 4)[0]
initial_idx = np.concatenate((birad1_ind[-250:], birad2_ind[-250:], birad3_ind[-250:], birad4_ind[-250:], birad5_ind[-120:]))


X_initial = X_train[initial_idx]
y_initial = ybi_train[initial_idx]

#%%
# generate the pool
# remove the initial data from the training dataset
X_pool = np.delete(X_train, initial_idx, axis=0)
y_pool = np.delete(ybi_train, initial_idx, axis=0)

#%%
# initializing the learner

learner = ActiveLearner(
    estimator= lgb.LGBMClassifier(application = "multiclass",  learning_rate = 0.11, num_iterations=70, num_leaves = 100,\
     boosting_type="goss", max_depth = 6, max_bin = 200,\
     task ="train", objective = "multiclass", num_classes =5, seed = 100, data_random_seed = 100, bagging_seed = 10, \
      is_unbalance = True) ,
    X_training=X_initial, y_training=y_initial, verbose=1, query_strategy=uncertainty_batch_sampling
) 
y_pred=learner.predict(RX_test)
maxSrc = f1_score(Rybi_test, y_pred, average = 'macro')
print("maxSrc = ", maxSrc)
#holdout
print("_________HOLDOUT_________")
print("LEFT BIRAD:")
PrintResult(learner, LX_hold, Lybi_hold )
print("RIGHT BIRAD:")
PrintResult(learner, RX_hold, Rybi_hold )

#STUDY
L_bi = learner.predict(LX_hold)
R_bi = learner.predict(RX_hold)
#L_den = clfden.predict(LX_ho)
#R_den = clfden.predict(RX_ho)
ybi = getmax(list(Lybi_ho), list(Rybi_ho))
#yden = getmax(list(Lyden_ho), list(Ryden_ho))
ybi_pred = getmax(L_bi, R_bi)
#yden_pred = getmax(L_den, R_den)

print("____BIRAD___")
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(ybi, ybi_pred)))
cm = confusion_matrix(ybi, ybi_pred)
print('Confusion matrix\n\n', cm)

print(classification_report(ybi, ybi_pred))
print("F1 macro:", f1_score(ybi, ybi_pred, average = 'macro'))

filename = '/home/sam/Mammography/code/modelLGBM/Toplearner_Left_512-avg.sav'
pickle.dump(learner, open(filename, 'wb'))

"""
# query for labels
# the active learning loop
n_queries_limit = 200
count = 0
for idx in tqdm(range(n_queries_limit)):
    query_idx, query_instance = learner.query(X_pool, n_instances=5)
    learner.teach(X_pool[query_idx], y_pool[query_idx])
    # remove queried instance from pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)
    y_pred=learner.predict(RX_test)
    scr = f1_score(Rybi_test, y_pred, average = 'macro')
    y_pred_ho=learner.predict(RX_hold)
    scr_ho = f1_score(Rybi_hold, y_pred_ho, average = 'macro')
    print("f1-score for holdout: ", scr_ho)
    if scr > maxSrc:
        print("Save: ", maxSrc, " --> ",scr )
        maxSrc = scr
        count = 0
        #for Top Learner
        print("F1 macro:", scr)
        filename = '/home/sam/Mammography/code/modelLGBM/Toplearner_Right_512-avg-top.sav'
        pickle.dump(learner, open(filename, 'wb'))
    else:
        count += 1
        print("Not improve: count = ", count)
    if count == 30:
        print("F1 macro:", scr)
        filename = '/home/sam/Mammography/code/modelLGBM/Toplearner_Right_512-avg-last.sav'
        pickle.dump(learner, open(filename, 'wb'))
        break


RTopLearner = pickle.load(open('/home/sam/Mammography/code/modelLGBM/Toplearner_Right_512-avg-last.sav', "rb"))
LTopLearner = pickle.load(open('/home/sam/Mammography/code/modelLGBM/Toplearner_Left_512-avg.sav', "rb"))

#holdout
print("_________HOLDOUT TOP LEARNER_________")
print("LEFT BIRAD:")
PrintResult(LTopLearner, LX_hold, Lybi_hold )

print("RIGHT BIRAD:")
PrintResult(RTopLearner, RX_hold, Rybi_hold )

#STUDY
L_bi = RTopLearner.predict(LX_hold)
R_bi = LTopLearner.predict(RX_hold)
#L_den = clfden.predict(LX_ho)
#R_den = clfden.predict(RX_ho)
ybi = getmax(list(Lybi_ho), list(Rybi_ho))
#yden = getmax(list(Lyden_ho), list(Ryden_ho))
ybi_pred = getmax(L_bi, R_bi)
#yden_pred = getmax(L_den, R_den)

print("____BIRAD___")
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(ybi, ybi_pred)))
cm = confusion_matrix(ybi, ybi_pred)
print('Confusion matrix\n\n', cm)

print(classification_report(ybi, ybi_pred))
print("F1 macro:", f1_score(ybi, ybi_pred, average = 'macro'))
#print("*** \n DENSITY:")
#PrintResult(clfden, X_hold, yden_hold )
"""
