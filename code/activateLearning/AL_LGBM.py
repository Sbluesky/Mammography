
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

#query stratery
def random_sampling(classifier, X_pool):
    n_samples = len(X_pool)
    query_idx = np.random.choice(range(n_samples))
    return query_idx, X_pool[query_idx]
#%%
df_train = pd.read_csv("/home/sam/Mammography/code/studyfeature_multitrain.csv", header =None)
df_test = pd.read_csv("/home/sam/Mammography/code/studyfeature_multivalid.csv", header =None)
df_holdout = pd.read_csv("/home/sam/Mammography/code/studyfeature_multiholdout.csv", header =None)

#%%
#tempLCC, tempLMLO, tempRCC, tempRMLO, templbi, temprbi,templden,temprden,tempbi,tempden: feature format [512,1] 
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
#%%
#n_initial = 100
#initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
#birad 1->5: 3091 1731 658 311 66
birad1_ind = np.where(ybi_train == 1)[0] #return tuple
birad2_ind = np.where(ybi_train == 2)[0]
birad3_ind = np.where(ybi_train == 3)[0]
birad4_ind = np.where(ybi_train == 4)[0]
birad5_ind = np.where(ybi_train == 5)[0]
initial_idx = np.concatenate((birad1_ind[:100], birad2_ind[:100], birad3_ind[:100], birad4_ind[:100], birad5_ind[:50]))


X_initial = X_train[initial_idx]
y_initial = ybi_train[initial_idx]

#%%
# generate the pool
# remove the initial data from the training dataset
X_pool = np.delete(X_train, initial_idx, axis=0)
y_pool = np.delete(ybi_train, initial_idx, axis=0)

#%%
# initializing the learner
"""
learner = ActiveLearner(
    estimator= lgb.LGBMClassifier(application = "multiclass",  learning_rate = 0.1, num_iterations=80, num_leaves = 100,\
     boosting_type="goss", max_depth = 6, max_bin = 200,\
     task ="train", objective = "multiclass", num_classes =5,\
      is_unbalance = True) ,
    X_training=X_initial, y_training=y_initial, verbose=1, query_strategy=uncertainty_batch_sampling
) #class_weight={1:2, 2:2, 3:4, 4:5, 5:6} 
y_pred=learner.predict(X_hold)
maxSrc = f1_score(ybi_hold, y_pred, average = 'macro')
#holdout
print("_________HOLDOUT_________")
print("BIRAD:")
PrintResult(learner, X_hold, ybi_hold )

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
    y_pred=learner.predict(X_hold)
    scr = f1_score(ybi_hold, y_pred, average = 'macro')
    if scr > maxSrc:
        print("Save: ", maxSrc, " --> ",scr )
        maxSrc = scr
        count = 0
        #for Top Learner
        print("F1 macro:", scr)
        filename = '/home/sam/Mammography/code/modelLGBM/Toplearner_Uncertain_Batch_Samp.sav'
        pickle.dump(learner, open(filename, 'wb'))
    else:
        count += 1
        print("Not improve: count = ", count)
    if count == 50:
        print("F1 macro:", scr)
        break


#holdout
print("_________HOLDOUT_________")
print("BIRAD:")
PrintResult(learner, X_hold, ybi_hold )
"""
TopLearner = pickle.load(open('/home/sam/Mammography/code/modelLGBM/Toplearner_Uncertain_Batch_Samp.sav', "rb"))
#holdout
print("_________HOLDOUT TOP LEARNER_________")
print("BIRAD:")
PrintResult(TopLearner, X_hold, ybi_hold )


#print("*** \n DENSITY:")
#PrintResult(clfden, X_hold, yden_hold )

# save the model to disk
#filename = '/home/sam/Mammography/code/modelLGBM/Lastlearner_Entropy.sav'
#pickle.dump(learner, open(filename, 'wb'))

"""

learner = pickle.load(open("/home/sam/Mammography/code/modelLGBM/Lastlearner_Entropy.sav", "rb"))
print("score ", learner.score(X_test, ybi_test))
Toplearner = pickle.load(open("/home/sam/Mammography/code/modelLGBM/Toplearner_Entropy.sav", "rb"))
print("score ", Toplearner.score(X_test, ybi_test))
"""