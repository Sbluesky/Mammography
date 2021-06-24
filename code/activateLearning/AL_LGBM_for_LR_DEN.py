
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

def getvoting(L, R):
    res = []
    for i in range(len(L)):
        if (L[i] == 0 | R[i] ==0):
            res.append(0)
        elif (L[i]==1 | R[i] == 1):
            res.append(1)
        elif (L[i]==2 & R[i] == 2):
            res.append(2)
        else:
            res.append(3)

    return res
#%%
df_train = pd.read_csv("/home/sam/Mammography/code/beststudyfeature_multitrain.csv", header =None)
df_test = pd.read_csv("/home/sam/Mammography/code/beststudyfeature_multivalid.csv", header =None)
df_holdout = pd.read_csv("/home/sam/Mammography/code/beststudyfeature_multiholdout.csv", header =None)

df_train = getdataframe(df_train)
df_test = getdataframe(df_test)
df_holdout = getdataframe(df_holdout)
Ldf_holdout = df_holdout.iloc[0:1272,:]
Rdf_holdout = df_holdout.iloc[1272:,:]
Ldf_test = df_test.iloc[0:1267,:]
Rdf_test = df_test.iloc[1267:,:]
#%%
#tempLCC, tempLMLO, tempRCC, tempRMLO, templbi, temprbi,templden,temprden,tempbi,tempden: feature format [512,1] 

X_train = df_train.iloc[:, 0:1024]
yden_train = df_train.iloc[0:5884,1026].to_list() + df_train.iloc[5884:,1027].to_list()

LX_test = Ldf_test.iloc[:,0:1024]
RX_test = Rdf_test.iloc[:,0:1024]
Lyden_test = Ldf_test.iloc[:,1026] 
Ryden_test = Rdf_test.iloc[:,1027]


LX_ho = Ldf_holdout.iloc[:,0:1024] 
RX_ho = Rdf_holdout.iloc[:,0:1024]
Lyden_ho = Ldf_holdout.iloc[:,1026]
Ryden_ho = Rdf_holdout.iloc[:,1027]


# assemble initial data
"""
LX_train = LX_train.to_numpy()
Lybi_train = Lybi_train.to_numpy()
RX_train = RX_train.to_numpy()
Rybi_train = Rybi_train.to_numpy()
"""
X_train = X_train.to_numpy()
yden_train = np.array(yden_train)

LX_test = LX_test.to_numpy()
RX_test = RX_test.to_numpy()
Lyden_test = Lyden_test.to_numpy()
Ryden_test = Ryden_test.to_numpy()

LX_hold = LX_ho.to_numpy()
Lyden_hold = Lyden_ho.to_numpy()
RX_hold = RX_ho.to_numpy()
Ryden_hold = Ryden_ho.to_numpy()

#den0: 122, den1: 928, den2: 7117, den3: 3601
den0_ind = np.where(yden_train == 0)[0] #return tuple
den1_ind = np.where(yden_train == 1)[0]
den2_ind = np.where(yden_train == 2)[0]
den3_ind = np.where(yden_train == 3)[0]
initial_idx = np.concatenate((den0_ind[:122], den1_ind[:100], den1_ind[828:], den2_ind[:100], den2_ind[7000:], den3_ind[:100], den3_ind[3500:]))


X_initial = X_train[initial_idx]
y_initial = yden_train[initial_idx]

#%%
# generate the pool
# remove the initial data from the training dataset
X_pool = np.delete(X_train, initial_idx, axis=0)
y_pool = np.delete(yden_train, initial_idx, axis=0)

#%%
# initializing the learner

learner = ActiveLearner(
    estimator= lgb.LGBMClassifier(application = "multiclass",  learning_rate = 0.1, num_iterations=60, num_leaves = 100,\
     boosting_type="goss", max_depth = 6, max_bin = 200,\
     task ="train", objective = "multiclass", num_classes =4,\
      is_unbalance = True) ,
    X_training=X_initial, y_training=y_initial, verbose=1, query_strategy=margin_sampling
) #class_weight={1:2, 2:2, 3:4, 4:5, 5:6} 
y_pred=learner.predict(RX_test)
maxSrc = f1_score(Ryden_test, y_pred, average = 'macro')
print("maxSrc = ", maxSrc)
#holdout
print("_________HOLDOUT_________")
print("LEFT DENSITY:")
PrintResult(learner, LX_hold, Lyden_hold )
print("RIGHT BIRAD:")
PrintResult(learner, RX_hold, Ryden_hold )
# query for labels
# the active learning loop
"""
n_queries_limit = 200
count = 0
for idx in tqdm(range(n_queries_limit)):
    query_idx, query_instance = learner.query(X_pool, n_instances=10)
    learner.teach(X_pool[query_idx], y_pool[query_idx])
    # remove queried instance from pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)
    y_pred=learner.predict(RX_test)
    scr = f1_score(Ryden_test, y_pred, average = 'macro')
    if scr > maxSrc:
        print("Save: ", maxSrc, " --> ",scr )
        maxSrc = scr
        count = 0
        #for Top Learner
        print("F1 macro:", scr)
        filename = '/home/sam/Mammography/code/modelLGBM/Toplearner_Den_1024.sav'
        pickle.dump(learner, open(filename, 'wb'))
    else:
        count += 1
        print("Not improve: count = ", count)
    if count == 20:
        print("F1 macro:", scr)
        break


TopLearner = pickle.load(open('/home/sam/Mammography/code/modelLGBM/Toplearner_Den_1024.sav', "rb"))

#holdout
print("_________HOLDOUT TOP LEARNER_________")
print("LEFT DENSITY:")
PrintResult(TopLearner, LX_hold, Lyden_hold )

print("RIGHT DENSITY:")
PrintResult(TopLearner, RX_hold, Ryden_hold )
"""
#STUDY
L_den = learner.predict(LX_ho)
R_den = learner.predict(RX_ho)

yden = getvoting(list(Lyden_ho), list(Ryden_ho))
#yden = getmax(list(Lyden_ho), list(Ryden_ho))
yden_pred = getvoting(L_den, R_den)
#yden_pred = getmax(L_den, R_den)

print("____DENSITY___")
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(yden, yden_pred)))
cm = confusion_matrix(yden, yden_pred)
print('Confusion matrix\n\n', cm)

print(classification_report(yden, yden_pred))
print("F1 macro:", f1_score(yden, yden_pred, average = 'macro'))
#print("*** \n DENSITY:")
#PrintResult(clfden, X_hold, yden_hold )
filename = '/home/sam/Mammography/code/modelLGBM/Toplearner_Den_1024.sav'
pickle.dump(learner, open(filename, 'wb'))

