
#This file is used for TRAINING and TESTING in LGBM for studies
#%%
import pandas as pd 
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import lightgbm as lgb; 
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

#%%
df_train = pd.read_csv("/home/single4/mammo/mammo/data/updatedata/csv/studyfeature_multitrain.csv", header =None)
df_test = pd.read_csv("/home/single4/mammo/mammo/data/updatedata/csv/studyfeature_multivalid.csv", header =None)
df_holdout = pd.read_csv("/home/single4/mammo/mammo/data/updatedata/csv/studyfeature_multiholdout.csv", header =None)

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

#%%
clf = lgb.LGBMClassifier(application = "multiclass",  learning_rate = 0.05, num_iterations=6000,\
     boosting_type="dart", max_depth = 6, max_bin = 100,\
     task ="train", class_weight={1:1, 2:2, 3:4, 4:5, 5:6},\
     device = "gpu") 
#clfden = lgb.LGBMClassifier(application = "multiclass",  num_iterations=1000, boosting_type="goss", max_depth = 100,class_weight={0:8, 1:4, 2:2, 3:2})
clf.fit(X_train, ybi_train, verbose=True, eval_set= (X_test, ybi_test))
#clfden.fit(X_train, yden_train, verbose=True, eval_set = (X_test, yden_test))

#holdout
print("_________HOLDOUT_________")
print("BIRAD:")
PrintResult(clf, X_hold, ybi_hold )

#print("*** \n DENSITY:")
#PrintResult(clfden, X_hold, yden_hold )

# save the model to disk
filename = '/home/single4/mammo/mammo/sam/multiview/modelLGBM/studies2048_v2.sav'
pickle.dump(clf, open(filename, 'wb'))
#filename = '/home/single4/mammo/mammo/sam/multiview/modelLGBM/studies2048.sav'
#pickle.dump(clfden, open(filename, 'wb'))