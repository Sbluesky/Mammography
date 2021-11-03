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
        avg = pd.DataFrame(CCdf.iloc[:,:512].copy())
        for i in range(512):
                avg[avg.columns[i]] = (CCdf[CCdf.columns[i]] + MLOdf[MLOdf.columns[i]])/2
        return avg

def getview (df):
    CC = df.iloc[:,0:512]
    MLO = df.iloc[:,512:1024]
    MLO.columns = range(MLO.shape[1])
    return CC,MLO

#tempLCC [0:512], tempLMLO[512:1024], tempRCC [1024:1536], tempRMLO[1536:2048], templbi[2048], temprbi[2049],templden[2050],temprden[2051],tempbi[2052],tempden[2053]: feature format [512,1] 
def getdataframe(df):
    #function to get dataframe from studies
    Left = df.drop(df.columns[1024:], axis = 1)
    Left = pd.concat([Left, df[2048]], axis = 1)
    Left = pd.concat([Left, df[2050]], axis = 1)
    Left.columns = range(Left.shape[1])

    Right = df.drop(df.columns[0:1024], axis = 1)
    Right.columns = range(Right.shape[1])
    Right = Right.drop(Right.columns[1024:], axis = 1)
    Right = pd.concat([Right, df[2049]], axis = 1)
    Right = pd.concat([Right, df[2051]], axis = 1)
    Right.columns = range(Right.shape[1])

    #temp = pd.concat([Left, Right], axis = 0)
    return Left.reset_index().drop(columns=['index']), Right.reset_index().drop(columns=['index'])

df_train = pd.read_csv("/home/sam/Mammography/code/data/resnet34/beststudyfeature_multitrain.csv", header =None) 
df_test = pd.read_csv("/home/sam/Mammography/code/data/resnet34/beststudyfeature_multivalid.csv", header =None) 
df_holdout = pd.read_csv("/home/sam/Mammography/code/data/resnet34/beststudyfeature_multiholdout.csv", header =None) 

#%%
Ldf_train, Rdf_train = getdataframe(df_train)
Ldf_test, Rdf_test = getdataframe(df_test)
Ldf_holdout, Rdf_holdout = getdataframe(df_holdout)


#%%
Ldf_trainCC, Ldf_trainMLO = getview(Ldf_train) #5884 rows, 514 columns
Rdf_trainCC, Rdf_trainMLO = getview(Rdf_train)

Ldf_testCC, Ldf_testMLO = getview(Ldf_test)
Rdf_testCC, Rdf_testMLO = getview(Rdf_test)

Ldf_hoCC, Ldf_hoMLO = getview(Ldf_holdout)
Rdf_hoCC, Rdf_hoMLO = getview(Rdf_holdout)


# %%

X_train = pd.concat((avgfeature(Ldf_trainCC,Ldf_trainMLO), avgfeature(Rdf_trainCC,Rdf_trainMLO)), axis = 0).to_numpy()
ybi_train = list(Ldf_train[1024]) + list(Rdf_train[1024])
ybi_train = np.array(ybi_train)
yden_train = list(Ldf_train[1025]) + list(Rdf_train[1025])

#%%

X_test = pd.concat((avgfeature(Ldf_testCC,Ldf_testMLO), avgfeature(Rdf_testCC,Rdf_testMLO)), axis = 0)
ybi_test = list(Ldf_test[1024]) + list(Rdf_test[1024])
yden_test = list(Ldf_test[1025]) + list(Rdf_test[1025])


LX_ho =  avgfeature(Ldf_hoCC,Ldf_hoMLO)
RX_ho = avgfeature(Rdf_hoCC,Rdf_hoMLO)

Lybi_ho = list(Ldf_holdout[1024])
Rybi_ho = list(Rdf_holdout[1024])
Lyden_ho = list(Ldf_holdout[1025])
Ryden_ho = list(Rdf_holdout[1025])


birad1_ind = np.where(ybi_train == 0)[0] #return tuple
birad2_ind = np.where(ybi_train == 1)[0]
birad3_ind = np.where(ybi_train == 2)[0]
birad4_ind = np.where(ybi_train == 3)[0]
birad5_ind = np.where(ybi_train == 4)[0]
initial_idx = np.concatenate((birad1_ind[-500:], birad2_ind[-500:], birad3_ind[-500:], birad4_ind[-500:], birad5_ind[-500:]))

X_initial = X_train[initial_idx]
y_initial = ybi_train[initial_idx]

# %%
# build the lightgbm model
"""
clfbi = lgbm.LGBMClassifier(application = "multiclass", learning_rate = 0.09, num_iterations=23, num_leaves = 100,\
boosting_type="goss", max_depth = 6, max_bin = 200,\
task ="train", objective = "multiclass", num_classes =5, seed = 100, data_random_seed = 100, bagging_seed = 100, \
is_unbalance = True) 
clfbi.fit(X_initial, y_initial, verbose=True, eval_set= (X_test, ybi_test))
"""
clfden = lgbm.LGBMClassifier(application = "multiclass", learning_rate = 0.09, num_iterations=200, num_leaves = 100,\
boosting_type="goss", max_depth = 4, max_bin = 200,\
task ="train", objective = "multiclass", num_classes =4, seed = 200, data_random_seed = 200, bagging_seed = 200, class_weight={0:7, 1:3, 2:4, 3:4} ,\
is_unbalance = True, first_metric_only = True) 
clfden.fit(X_train, yden_train, verbose=True, eval_set = (X_test, yden_test) ,eval_metric = 'auc_mu', early_stopping_rounds = 100)



filenameden = '/home/sam/Mammography/code/modelLGBM/resnet34/LGBM_512_avg_DENSITY.sav'
#pickle.dump(clfbi, open(filenamebi, 'wb'))
pickle.dump(clfden, open(filenameden, 'wb'))

#Lcls = pickle.load(open('/home/sam/Mammography/code/modelLGBM/resnet34/LGBM_512_avg_top_left.sav', "rb"))
#Rcls = pickle.load(open('/home/sam/Mammography/code/modelLGBM/resnet34/LGBM_512_avg_top_right.sav', "rb"))
#clfden = pickle.load(open(filenameden, 'rb'))

print("_________HOLDOUT_________")
print("__LEFT__")
"""
print("BIRAD:")
Lybi_pred = Lcls.predict(LX_ho)
print("len Lypred: " ,len(Lybi_pred))
print("len Lybiho: ", len(Lybi_ho))
PrintResult( Lybi_pred, Lybi_ho )
"""
print("*** \n DENSITY:")
Lyden_pred = clfden.predict(LX_ho)
PrintResult( Lyden_pred, Lyden_ho )


print("__RIGHT__")
"""
print("BIRAD:")
Rybi_pred = Rcls.predict(RX_ho)
PrintResult( Rybi_pred, Rybi_ho )
"""
print("*** \n DENSITY:")
Ryden_pred = clfden.predict(RX_ho)
PrintResult( Ryden_pred, Ryden_ho )

#STUDY
#%%

#ybi = getmax(list(Lybi_ho), list(Rybi_ho))
yden = getmax(list(Lyden_ho), list(Ryden_ho))
#ybi_pred = getmax(Lybi_pred, Rybi_pred)
yden_pred = getmax(Lyden_pred, Ryden_pred)
print('-------STUDY-------')
"""
print("____BIRAD___")
PrintResult( ybi_pred, ybi )
"""
print("____DENSITY___")
PrintResult( yden_pred, yden )


print("__SIDE__")
print("___DENSITY___")
yside = list(Lyden_ho) + list(Ryden_ho)
yside_pred = list(Lyden_pred) + list(Ryden_pred)
PrintResult(yside_pred, yside)




# %%
