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
                avg[avg.columns[i]] = (CCdf[CCdf.columns[i]] + MLOdf[MLOdf.columns[i]])
        return avg

def getview (df):
    return df[df.index%2==0].reset_index(drop=True), df[df.index%2!=0].reset_index(drop=True)

df_train = pd.read_csv("/home/sam/Mammography/code/data/resnet34/singlestudytrain.csv", header =None) #23536 rows, 514 columns
df_test = pd.read_csv("/home/sam/Mammography/code/data/resnet34/singlestudyvalid.csv", header =None) #1267 rows
df_holdout = pd.read_csv("/home/sam/Mammography/code/data/resnet34/singlestudyholdout.csv", header =None) #1272 rows

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
ybi_train = getmax(list(Ldf_trainCC.iloc[:,512]), list(Ldf_trainMLO.iloc[:,512])) + getmax(list(Rdf_trainCC.iloc[:,512]), list(Rdf_trainMLO.iloc[:,512]))
ybi_train = np.array(ybi_train)
yden_train = getmax(list(Ldf_trainCC.iloc[:,513]), list(Ldf_trainMLO.iloc[:,513])) + getmax(list(Rdf_trainCC.iloc[:,513]), list(Rdf_trainMLO.iloc[:,513]))

#%%

X_test = pd.concat((avgfeature(Ldf_testCC,Ldf_testMLO), avgfeature(Rdf_testCC,Rdf_testMLO)), axis = 0)
ybi_test = getmax(list(Ldf_testCC.iloc[:,512]), list(Ldf_testMLO.iloc[:,512])) + getmax(list(Rdf_testCC.iloc[:,512]), list(Rdf_testMLO.iloc[:,512]))
yden_test = getmax(list(Ldf_testCC.iloc[:,513]), list(Ldf_testMLO.iloc[:,513])) + getmax(list(Rdf_testCC.iloc[:,513]), list(Rdf_testMLO.iloc[:,513]))



LX_ho =  avgfeature(Ldf_hoCC,Ldf_hoMLO)
RX_ho = avgfeature(Rdf_hoCC,Rdf_hoMLO)

Lybi_ho = getmax(list(Ldf_hoCC.iloc[:,512]), list(Ldf_hoMLO.iloc[:,512]))
Rybi_ho = getmax(list(Rdf_hoCC.iloc[:,512]), list(Rdf_hoMLO.iloc[:,512]))
Lyden_ho = getmax(list(Ldf_hoCC.iloc[:,513]), list(Ldf_hoMLO.iloc[:,513]))
Ryden_ho = getmax(list(Rdf_hoCC.iloc[:,513]), list(Rdf_hoMLO.iloc[:,513]))


birad1_ind = np.where(ybi_train == 1)[0] #return tuple
birad2_ind = np.where(ybi_train == 2)[0]
birad3_ind = np.where(ybi_train == 3)[0]
birad4_ind = np.where(ybi_train == 4)[0]
birad5_ind = np.where(ybi_train == 5)[0]
initial_idx = np.concatenate((birad1_ind[-500:], birad2_ind[-500:], birad3_ind[-500:], birad4_ind[-500:], birad5_ind))

X_initial = X_train[initial_idx]
y_initial = ybi_train[initial_idx]
# %%
# build the lightgbm model
"""
clfbi = lgbm.LGBMClassifier(application = "multiclass", learning_rate = 0.1, num_iterations=39, num_leaves = 100,\
boosting_type="goss", max_depth = 6, max_bin = 200,\
task ="train", objective = "multiclass", num_classes =5, seed = 10, data_random_seed = 100, bagging_seed = 100, \
is_unbalance = True)
clfden = lgbm.LGBMClassifier(application = "multiclass",  num_iterations=80, boosting_type="dart", max_depth = 6, max_bin=200,class_weight={0:3, 1:2, 2:1, 3:1} )
clfbi.fit(X_initial, y_initial, verbose=True, eval_set= (X_test, ybi_test))
clfden.fit(X_train, yden_train, verbose=True, eval_set = (X_test, yden_test))
"""


filenamebi = '/home/sam/Mammography/code/modelLGBM/resnet34/LGBM_1model_avg_birad.sav'
filenameden = '/home/sam/Mammography/code/modelLGBM/resnet34/LGBM_1model_avg_density.sav'
#pickle.dump(clfbi, open(filenamebi, 'wb'))
#pickle.dump(clfden, open(filenameden, 'wb'))

clfbi = pickle.load(open(filenamebi, 'rb'))
clfden = pickle.load(open(filenameden, 'rb'))

print("_________HOLDOUT_________")
print("__LEFT__")
print("BIRAD:")
Lybi_pred = clfbi.predict(LX_ho)
PrintResult( Lybi_pred, Lybi_ho )

print("*** \n DENSITY:")
Lyden_pred = clfden.predict(LX_ho)
PrintResult( Lyden_pred, Lyden_ho )


print("__RIGHT__")
print("BIRAD:")
Rybi_pred = clfbi.predict(RX_ho)
PrintResult( Rybi_pred, Rybi_ho )

print("*** \n DENSITY:")
Ryden_pred = clfden.predict(RX_ho)
PrintResult( Ryden_pred, Ryden_ho )

#STUDY
#%%

ybi = getmax(list(Lybi_ho), list(Rybi_ho))
yden = getmax(list(Lyden_ho), list(Ryden_ho))
ybi_pred = getmax(Lybi_pred, Rybi_pred)
yden_pred = getmax(Lyden_pred, Ryden_pred)
print('-------STUDY-------')
print("____BIRAD___")
PrintResult( ybi_pred, ybi )

print("____DENSITY___")
PrintResult( yden_pred, yden )


print("__SIDE__")
print("___DENSITY___")
yside = list(Lyden_ho) + list(Ryden_ho)
yside_pred = list(Lyden_pred) + list(Ryden_pred)
PrintResult(yside_pred, yside)




# %%
