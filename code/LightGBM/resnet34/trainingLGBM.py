#%%
from os import device_encoding
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
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

def getmax(L, R):
    res = []
    for i in range(len(L)):
        res.append(max(L[i], R[i]))
    return res
#%%
def getdataframe(df):
    #function to get dataframe from studies
    Left = df.drop(df.columns[1024:2048], axis = 1)
    Left.columns = range(Left.shape[1])
    Right = df.drop(df.columns[0:1024], axis = 1)
    Right.columns = range(Right.shape[1])
    temp = pd.concat([Left, Right], axis = 0, )
    return temp.reset_index().drop(columns=['index'])


#tempLCC, tempLMLO, tempRCC, tempRMLO, templbi, temprbi,templden,temprden,tempbi,tempden: feature format [512,1] 
df_train = pd.read_csv("/home/single4/mammo/mammo/data/updatedata/csv/beststudyfeature_multitrain.csv", header =None) #5884 rows
df_test = pd.read_csv("/home/single4/mammo/mammo/data/updatedata/csv/beststudyfeature_multivalid.csv", header =None) #1267 rows
df_holdout = pd.read_csv("/home/single4/mammo/mammo/data/updatedata/csv/beststudyfeature_multiholdout.csv", header =None) #1272 rows
df_train = getdataframe(df_train)
df_test = getdataframe(df_test)
df_holdout = getdataframe(df_holdout)
Ldf_holdout = df_holdout.iloc[0:1272,:]
Rdf_holdout = df_holdout.iloc[1272:,:]

# %%
X_train = df_train.iloc[:, 0:1024]
ybi_train = df_train.iloc[0:5884,1024].to_list() + df_train.iloc[5884:,1025].to_list() #concat label of Left Birad [0:5884,1024] và Right Birad [5884:,1025]
yden_train = df_train.iloc[0:5884,1026].to_list() + df_train.iloc[5884:,1027].to_list()

X_test = df_test.iloc[:,0:1024]
ybi_test = df_test.iloc[0:1267,1024].to_list() + df_test.iloc[1267:,1025].to_list()
yden_test = df_test.iloc[0:1267,1026].to_list() + df_test.iloc[1267:,1027].to_list()


LX_ho = Ldf_holdout.iloc[:,0:1024] 
RX_ho = Rdf_holdout.iloc[:,0:1024]

Lybi_ho = Ldf_holdout.iloc[:,1024] + 1
Rybi_ho = Rdf_holdout.iloc[:,1025]
Lyden_ho = Ldf_holdout.iloc[:,1026]
Ryden_ho = Rdf_holdout.iloc[:,1027]


#print(X_train.isnull().values.any()) #check null
"""
#gridSearch
from sklearn.model_selection import GridSearchCV
param_test1 = {
 'max_depth':[4,5,6,7,8],
 'num_iterations':range(60,181,20)
}

gsearch1 = GridSearchCV(estimator = lgb.LGBMClassifier(application = "multiclass", num_iterations=110, boosting_type="goss", max_depth = 6, max_bin=200, class_weight={0:1, 1:2, 2:6, 3:7, 4:5}, task ="train", device = "gpu"), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4, cv=5)

gsearch1.fit(X_train, ybi_train)
print(gsearch1.param_grid, gsearch1.best_params_, gsearch1.best_score_)
"""
# %%
# build the lightgbm model
clfL = pickle.load(open('/home/single4/mammo/mammo/sam/multiview/modelLGBM/Bi_512_lgbmtop.sav', 'rb'))
#clfden = pickle.load(open('/home/single4/mammo/mammo/huyen/cls_multi/model/Density_lgbmstacking_weight8423.sav', 'rb'))
#clfL = lgb.LGBMClassifier(application = "multiclass", num_iterations=300, learning_rate = 0.1, boosting_type="gbdt", max_depth = 6, max_bin=200, class_weight={0:0.5, 1:2, 2:7, 3:3, 4:5}, task ="train", device = "gpu")
#clfden = lgb.LGBMClassifier(application = "multiclass",  num_iterations=2000, boosting_type="dart", max_depth = 6, max_bin=200, class_weight={0:5, 1:4, 2:1, 3:2}, device = "gpu")
#clfL.fit(df_train.iloc[0:5884, 0:1024], df_train.iloc[0:5884,1024], verbose=True, eval_set= (df_test.iloc[0:1267,0:1024], df_test.iloc[0:1267,1024]))
#clfden.fit(X_train, yden_train, verbose=True, eval_set = (X_test, yden_test))

clfR = lgb.LGBMClassifier(application = "multiclass", num_iterations=160, boosting_type="gbdt", max_depth = 7, max_bin=200, class_weight={0:1, 1:2, 2:6, 3:10, 4:5}, task ="train", device = "gpu")
clfR.fit(df_train.iloc[5884:, 0:1024], df_train.iloc[5884:,1025], verbose=True, eval_set= (df_test.iloc[1267:,0:1024], df_test.iloc[1267:,1024]))


#holdout
print("_________HOLDOUT_________")
print("__LEFT__")
print("BIRAD:")
PrintResult(clfL, LX_ho, Lybi_ho )

#print("*** \n DENSITY:")
#PrintResult(clfden, LX_ho, Lyden_ho )

print("__RIGHT__")
print("BIRAD:")
PrintResult(clfR, RX_ho, Rybi_ho )

#print("*** \n DENSITY:")
#PrintResult(clfden, RX_ho, Ryden_ho )

#STUDY
L_bi = clfL.predict(LX_ho)
R_bi = clfR.predict(RX_ho)
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

"""
print("____DENSITY___")
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(yden, yden_pred)))
cm = confusion_matrix(yden, yden_pred)
print('Confusion matrix\n\n', cm)

print(classification_report(yden, yden_pred))
print("F1 macro:", f1_score(yden, yden_pred, average = 'macro'))
"""

# save the model to disk
#filename = '/home/single4/mammo/mammo/sam/multiview/modelLGBM/LGBM_1024_Right.sav'
#pickle.dump(clfR, open(filename, 'wb'))
#filename = '/home/single4/mammo/mammo/sam/multiview/modelLGBM/Den_512_lgbmtop1.sav'
#pickle.dump(clfden, open(filename, 'wb'))

#%%
"""
# %%
# check the distribution of the target variable
df_train['Birad'].value_counts()
df_train['Density'].value_counts()
#Birad 1: 7993, Birad 2: 2568, Birad 3: 817, Birad 4: 323, Birad 5: 67
#Den0: 122 Den1: 928 Den2: 7117 Den3: 3601
# %%
Xbi_train = df_train[['Bi_CC','Bi_MLO']]
Xden_train = df_train[['Den_CC','Den_MLO']]
ybi_train = df_train['Birad']
yden_train = df_train["Density"]

Xbi_test = df_test[['Bi_CC','Bi_MLO']]
Xden_test = df_test[['Den_CC','Den_MLO']]
ybi_test = df_test['Birad']
yden_test = df_test["Density"]
    

LXbi_ho = Ldf_holdout[['Bi_CC','Bi_MLO']]
RXbi_ho = Rdf_holdout[['Bi_CC','Bi_MLO']]
LXden_ho = Ldf_holdout[['Den_CC','Den_MLO']]
RXden_ho = Rdf_holdout[['Den_CC','Den_MLO']]

Lybi_ho = Ldf_holdout['Birad']
Rybi_ho = Rdf_holdout['Birad']
Lyden_ho = Ldf_holdout["Density"]
Ryden_ho = Rdf_holdout["Density"]
# %%
# build the lightgbm mod'el
clf = lgb.LGBMClassifier(application = "multiclass",  num_iterations=30, boosting_type="goss", max_depth = 2, max_bin=1000)
clfden = lgb.LGBMClassifier(application = "multiclass",  num_iterations=30, boosting_type="goss", max_depth = 2)
clf.fit(Xbi_train, ybi_train)
clfden.fit(Xden_train, yden_train)

# view accuracy
print("_________TEST_________")
print("BIRAD:")
PrintResult(clf, Xbi_test, ybi_test )

print("*** \n DENSITY:")
PrintResult(clfden, Xden_test, yden_test )

#holdout
print("_________HOLDOUT_________")
print("__LEFT__")
print("BIRAD:")
PrintResult(clf, LXbi_ho, Lybi_ho )

print("*** \n DENSITY:")
PrintResult(clfden, LXden_ho, Lyden_ho )

print("__RIGHT__")
print("BIRAD:")
PrintResult(clf, RXbi_ho, Rybi_ho )

print("*** \n DENSITY:")
PrintResult(clfden, RXden_ho, Ryden_ho )

# save the model to disk
filename = '/home/single4/mammo/mammo/sam/multiview/modelLGBM/Bi_lgbmtop.sav'
pickle.dump(clf, open(filename, 'wb'))
filename = '/home/single4/mammo/mammo/sam/multiview/modelLGBM/Den_lgbmtop.sav'
pickle.dump(clfden, open(filename, 'wb'))
# some time later...

# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)
"""