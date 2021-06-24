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

df_train = pd.read_csv("/home/sam/Mammography/code/data/singlestudytrain.csv", header =None) #23536 rows, 514 columns
df_test = pd.read_csv("/home/sam/Mammography/code/data/singlestudyvalid.csv", header =None) #1267 rows
df_holdout = pd.read_csv("/home/sam/Mammography/code/data/singlestudyholdout.csv", header =None) #1272 rows
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


LX_ho = avgfeature(Ldf_hoCC,Ldf_hoMLO)
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
initial_idx = np.concatenate((birad1_ind[-250:], birad2_ind[-250:], birad3_ind[-250:], birad4_ind[-250:], birad5_ind))

X_initial = X_train[initial_idx]
y_initial = ybi_train[initial_idx]
# %%
# build the lightgbm model

clfbi = lgbm.LGBMClassifier(application = "multiclass", learning_rate = 0.1, num_iterations=39, num_leaves = 100,\
boosting_type="goss", max_depth = 6, max_bin = 200,\
task ="train", objective = "multiclass", num_classes =5, seed = 10, data_random_seed = 100, bagging_seed = 100, \
is_unbalance = True)
#clfden = lgbm.LGBMClassifier(application = "multiclass",  num_iterations=80, boosting_type="dart", max_depth = 6, max_bin=200)
clfbi.fit(X_initial, y_initial, verbose=True, eval_set= (X_test, ybi_test))
#clfden.fit(X_train, yden_train, verbose=True, eval_set = (X_test, yden_test))

filenamebi = '/home/sam/Mammography/code/modelLGBM/LGBM_1model_avg_birad.sav'
pickle.dump(clfbi, open(filenamebi, 'wb'))
filenameden = '/home/sam/Mammography/code/modelLGBM/LGBM_1model_avg_density.sav'
#pickle.dump(clfden, open(filenameden, 'wb'))

"""
print("_________HOLDOUT_________")
print("BIRAD:")
PrintResult(clfbi, X_ho, ybi_ho )

print("*** \n DENSITY:")
PrintResult(clfden, X_ho, yden_ho )



"""
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




"""
#holdout
LX_ho_CC,LX_ho_MLO = getview(LX_ho)
Lybi_ho_CC, Lybi_ho_MLO = getview(Lybi_ho)
Lyden_ho_CC,Lyden_ho_MLO = getview(Lyden_ho)

RX_ho_CC,RX_ho_MLO = getview(RX_ho)
Rybi_ho_CC, Rybi_ho_MLO = getview(Rybi_ho)
Ryden_ho_CC,Ryden_ho_MLO = getview(Ryden_ho)

print("_________HOLDOUT_________")
print("__LEFT__")
print("BIRAD:")
y_pred_CC = clfbi.predict(LX_ho_CC)
y_pred_MLO = clfbi.predict(LX_ho_MLO)
Lybi_pred = getmax(y_pred_CC,y_pred_MLO)
Lybi = getmax(list(Lybi_ho_CC),list(Lybi_ho_MLO))
PrintResult( Lybi_pred, Lybi )

print("*** \n DENSITY:")
y_pred_CC = clfden.predict(LX_ho_CC)
y_pred_MLO = clfden.predict(LX_ho_MLO)
Lyden_pred = getmax(y_pred_CC,y_pred_MLO)
Lyden = getmax(list(Lyden_ho_CC),list(Lyden_ho_MLO))
PrintResult( Lyden_pred, Lyden )


print("__RIGHT__")
print("BIRAD:")
y_pred_CC = clfbi.predict(RX_ho_CC)
y_pred_MLO = clfbi.predict(RX_ho_MLO)
Rybi_pred = getmax(y_pred_CC,y_pred_MLO)
Rybi = getmax(list(Rybi_ho_CC),list(Rybi_ho_MLO))
PrintResult( Rybi_pred, Rybi )
print("*** \n DENSITY:")
y_pred_CC = clfden.predict(RX_ho_CC)
y_pred_MLO = clfden.predict(RX_ho_MLO)
Ryden_pred = getmax(y_pred_CC,y_pred_MLO)
Ryden = getmax(list(Ryden_ho_CC),list(Ryden_ho_MLO))
PrintResult( Ryden_pred, Ryden )

#STUDY
#%%

ybi = []
for i in range(0,df_holdout.shape[0],4):
    ybi.append(max(df_holdout.iloc[i,512],df_holdout.iloc[i+1,512],df_holdout.iloc[i+2,512],df_holdout.iloc[i+3,512]))
print("birad 5:", ybi.count(5))

#ybi = getmax(list(Lybi), list(Rybi))
yden = getmax(list(Lyden), list(Ryden))
ybi_pred = getmax(Lybi_pred, Rybi_pred)
yden_pred = getmax(Lyden_pred, Ryden_pred)
print('-------STUDY-------')
print("____BIRAD___")
PrintResult( ybi_pred, ybi )


print("____DENSITY___")
PrintResult( yden_pred, yden )
# save the model to disk
#filename = '/home/single4/mammo/mammo/sam/multiview/modelLGBM/LGBM_1024_Right.sav'
#pickle.dump(clfR, open(filename, 'wb'))
#filename = '/home/single4/mammo/mammo/sam/multiview/modelLGBM/Den_512_lgbmtop1.sav'
#pickle.dump(clfden, open(filename, 'wb'))

"""
# %%
