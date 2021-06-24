#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner, Committee
import lightgbm as lgb;
import pandas as pd
from tqdm import tqdm
import pickle 
from modAL.batch import uncertainty_batch_sampling
from modAL.uncertainty import uncertainty_sampling, entropy_sampling,margin_sampling

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

def get_trainindex(y_train, n_learner = 0):
    birad1_ind = np.where(y_train == 1)[0] #return tuple
    birad2_ind = np.where(y_train == 2)[0]
    birad3_ind = np.where(y_train == 3)[0]
    birad4_ind = np.where(y_train == 4)[0]
    birad5_ind = np.where(y_train == 5)[0]
    start = 0
    end = 100
    initial_idx = np.concatenate((birad1_ind[start:end], birad2_ind[start:end], birad3_ind[start:end], birad4_ind[start:end], birad5_ind[:25]))
    return initial_idx

#%%
df_train = pd.read_csv("/home/sam/Mammography/code/studyfeature_multitrain.csv", header =None)
df_test = pd.read_csv("/home/sam/Mammography/code/studyfeature_multivalid.csv", header =None)
df_holdout = pd.read_csv("/home/sam/Mammography/code/studyfeature_multiholdout.csv", header =None)

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

X_pool = X_train
y_pool = ybi_train
"""
# visualizing the classes
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(7, 7))
    pca = PCA(n_components=2).fit_transform(X_hold)
    plt.scatter(x=pca[:, 0], y=pca[:, 1], c=ybi_hold, cmap='viridis', s=50)
    plt.title('The iris dataset')
    plt.show()

"""
# initializing Committee members
n_members = 2
learner_list = list()

for member_idx in range(n_members):
    # initial training data
    train_idx = get_trainindex(ybi_train, n_learner = member_idx)
    X_train = X_pool[train_idx]
    y_train = y_pool[train_idx]

    # creating a reduced copy of the data with the known instances removed
    X_pool = np.delete(X_pool, train_idx, axis=0)
    y_pool = np.delete(y_pool, train_idx)

    if member_idx%2 == 0:
        qr_str = margin_sampling
    else:
        qr_str = entropy_sampling
    # initializing learner
    learner = ActiveLearner(
        estimator=RandomForestClassifier(),
        X_training=X_train, y_training=y_train, query_strategy = qr_str
    )
    learner_list.append(learner)

# assembling the committee
committee = Committee(learner_list=learner_list)

"""
# visualizing the Committee's predictions per learner

with plt.style.context('seaborn-white'):
    plt.figure(figsize=(n_members*7, 7))
    for learner_idx, learner in enumerate(committee):
        plt.subplot(1, n_members, learner_idx + 1)
        plt.scatter(x=pca[:, 0], y=pca[:, 1], c=learner.predict(X_hold), cmap='viridis', s=50)
        plt.title('Learner no. %d initial predictions' % (learner_idx + 1))
    plt.show()

# visualizing the initial predictions
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(7, 7))
    prediction = committee.predict(X_hold)
    plt.scatter(x=pca[:, 0], y=pca[:, 1], c=prediction, cmap='viridis', s=50)
    plt.title('Committee initial predictions, accuracy = %1.3f' % committee.score(X_hold, ybi_hold))
    plt.show()
"""
y_pred = committee.predict(X_hold)
maxSrc = f1_score(ybi_hold, y_pred, average = "macro")
n_queries_limit = 200
count = 0
for idx in tqdm(range(n_queries_limit)):
    query_idx, query_instance = committee.query(X_pool)
    committee.teach(
            X=X_pool[query_idx].reshape(1, -1),
            y=y_pool[query_idx].reshape(1, )
        )
    # remove queried instance from pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)
    y_pred=committee.predict(X_hold)
    scr = f1_score(ybi_hold, y_pred, average = 'macro')
    if scr > maxSrc:
        print("Save: ", maxSrc, " --> ",scr )
        maxSrc = scr
        count = 0
        #for Top Learner
        print("F1 macro:", scr)
        filename = '/home/sam/Mammography/code/modelLGBM/Commitee.sav'
        pickle.dump(learner, open(filename, 'wb'))
    else:
        count += 1
        print("Not improve: count = ", count)
    if count == 50:
        print("F1 macro:", scr)
        break


"""
# visualizing the final predictions per learner
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(n_members*7, 7))
    for learner_idx, learner in enumerate(committee):
        plt.subplot(1, n_members, learner_idx + 1)
        plt.scatter(x=pca[:, 0], y=pca[:, 1], c=learner.predict(X_hold), cmap='viridis', s=50)
        plt.title('Learner no. %d predictions after %d queries' % (learner_idx + 1, n_queries))
    plt.show()

# visualizing the Committee's predictions
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(7, 7))
    prediction = committee.predict(X_hold)
    plt.scatter(x=pca[:, 0], y=pca[:, 1], c=prediction, cmap='viridis', s=50)
    plt.title('Committee predictions after %d queries, accuracy = %1.3f'
              % (n_queries, committee.score(X_hold, ybi_hold)))
    plt.show()
"""
#holdout
print("_________HOLDOUT AFTER QUERIES_________")
print("BIRAD:")
PrintResult(committee, X_hold, ybi_hold )