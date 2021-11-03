from numpy import true_divide
from xgboost import XGBClassifier 
import pandas as pd 
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import lightgbm as lgb; 
import pickle 
from sklearn.model_selection import GridSearchCV
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

eval_set = [(X_train, ybi_train), (X_test, ybi_test)]
eval_setden = [(X_train, yden_train), (X_test, yden_test)]

param_test1 = {
 'max_depth':range(3,10,1),
 'min_child_weight':range(1,6,1)
}

gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softmax', nthread=4, seed=27, tree_method = "gpu_hist",
 gpu_id = 0), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4, cv=5)

gsearch1.fit(X_train, ybi_train)
print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)

"""
#training model
bimodel = XGBClassifier(learning_rate =0.01,
 n_estimators=5000,
 max_depth=4,
 min_child_weight=6,
 gamma=0.1,
 subsample=0.7,
 colsample_bytree=0.7,
 objective= 'multi:softmax',
 nthread=4,
 seed=27,
 scale_pos_weight = 1,
 tree_method = "gpu_hist",
 gpu_id = 0
 ) 
bimodel.fit(X_train, ybi_train, eval_metric="auc", eval_set=eval_set, verbose=True)

denmodel = XGBClassifier(learning_rate =0.1,
 n_estimators=1000,
 max_depth=6,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 seed=27) 
denmodel.fit(X_train, yden_train, eval_metric="auc", eval_set=eval_setden, verbose=True)
"""
#prediction
#holdout
print("_________HOLDOUT_________")
print("BIRAD:")
PrintResult(bimodel, X_hold, ybi_hold )
"""
print("*** \n DENSITY:")
PrintResult(denmodel, X_hold, yden_hold )

filename = '/home/single4/mammo/mammo/sam/multiview/modelLGBM/XGB_Bi_2048_v1.sav'
pickle.dump(bimodel, open(filename, 'wb'))

filename = '/home/single4/mammo/mammo/sam/multiview/modelLGBM/XGB_Den_2048.sav'
pickle.dump(denmodel, open(filename, 'wb'))
#tree plot
from xgboost import plot_tree
import matplotlib.pyplot as plt
 
plt.figure(figsize=(19, 5))
plot_tree(bimodel)
plt.show()

# evaluation plot
# retrieve performance metrics 
results = bimodel.evals_result()
epochs = len(results['validation_0' ]['auc'])
x_axis = range(0, epochs)
 
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['auc'], label='Train')
ax.plot(x_axis, results['validation_1']['auc'], label='Test')
ax.legend()
plt.ylabel('AUC')
plt.title('XGBoost AUC')
plt.show()
"""