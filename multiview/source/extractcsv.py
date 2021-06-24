#%%
#Xử lý file holdout ở định dạng single view sang đánh giá multi view
import pandas as pd 
import numpy as np 
# %%
df = pd.read_csv("/home/single4/mammo/mammo/data/updatedata/csv/singleview-train.csv")
df.info()
#%%
df["max_birad"] = -1
df["max_density"] = -1

# %%
transdf = pd.DataFrame(columns = df.columns)
list_study = df["study_id"].unique()
#get max value of study
for std in list_study:
    temp_df = df[df["study_id"] == std]
    temp_df["max_birad"] = max(temp_df["label_birad"].values)
    temp_df["max_density"] = max(temp_df["label_density"].values)
    transdf = pd.concat((transdf,temp_df), axis = 0)

transdf.info()

#transdf.tail(12)
# %%
transdf.to_csv("/home/single4/mammo/mammo/data/updatedata/csv/multiview_train.csv", index = False)
# %%
df = pd.read_csv("/home/single4/mammo/mammo/data/updatedata/csv/multiview_train.csv")
df.info()
#%%
#xem co study nao duoi 4 view hay tren 4 view hay khong
df = pd.read_csv("/home/single4/mammo/mammo/data/updatedata/csv/multiview_train.csv")
list_study = df["study_id"].unique()
underview = []
overview = []
for std in list_study:
    temp_df = df[df["study_id"] == std]
    if len(temp_df) <4:
        underview.append(std)
    if len(temp_df) > 4:
        overview.append(std)
    
print(len(underview))
print(len(overview))

# %%
for std in underview:
    temp_df = df[df["study_id"] == std]
    print(temp_df["max_birad"])
    
# %%
index = []
for std in underview:
   index.append(df[df["study_id"] == std].index.values) 
#drop row co study_id < 4 view
for i in index:
    df.drop(i, axis = 0, inplace = True)
df.to_csv("/home/single4/mammo/mammo/data/updatedata/csv/multiview_train.csv", index = False)
# %%
df = pd.read_csv("/home/single4/mammo/mammo/data/updatedata/csv/multiview_train.csv")
df.info()
#tạo df muitiview
df_multiview = pd.DataFrame(columns = ["study_id", "L_CC_imid", "R_CC_imid", "L_MLO_imid", "R_MLO_imid","L_CC_birad", "L_CC_den", "L_MLO_bi", "L_MLO_den", "R_CC_birad", "R_CC_den", "R_MLO_bi", "R_MLO_den", "max_bi", "max_den"])
df_multiview["study_id"] = df["study_id"].unique()
list_study = df["study_id"].unique()
df_multiview.info()
#%%
for std in list_study:
    index = df_multiview[df_multiview["study_id"] == std].index
    temp_df = df[df["study_id"] == std]
    for i in range(4):
        if (temp_df.iloc[i,2] == "CC") & (temp_df.iloc[i,3] == "L"): #position & laterality
            df_multiview.iloc[index,1] = temp_df.iloc[i,1] #L-CC-imid= image-id
            df_multiview.iloc[index,5] = temp_df.iloc[i,7] #L-CC-bi = label_bi
            df_multiview.iloc[index,6] = temp_df.iloc[i,8] #L-CC-den = label-den
        elif (temp_df.iloc[i,2] == "CC") & (temp_df.iloc[i,3] == "R"): 
            df_multiview.iloc[index,2] = temp_df.iloc[i,1]
            df_multiview.iloc[index,9] = temp_df.iloc[i,7]
            df_multiview.iloc[index,10] = temp_df.iloc[i,8]
        elif (temp_df.iloc[i,2] == "MLO") & (temp_df.iloc[i,3] == "L"):
            df_multiview.iloc[index,3] = temp_df.iloc[i,1]
            df_multiview.iloc[index,7] = temp_df.iloc[i,7]
            df_multiview.iloc[index,8] = temp_df.iloc[i,8]
        elif (temp_df.iloc[i,2] == "MLO") & (temp_df.iloc[i,3] == "R"):
            df_multiview.iloc[index,4] = temp_df.iloc[i,1]
            df_multiview.iloc[index,11] = temp_df.iloc[i,7]
            df_multiview.iloc[index,12] = temp_df.iloc[i,8]
    df_multiview.iloc[index,13] = temp_df.iloc[0,9] #max birad
    df_multiview.iloc[index,14] = temp_df.iloc[0,10] #max density
# %%
df_multiview.info()
# %%
df_multiview.to_csv("/home/single4/mammo/mammo/data/updatedata/csv/multiview_train.csv", index = False)

# %%
import pandas as pd 
df = pd.read_csv("/home/single4/mammo/mammo/data/updatedata/csv/multiview_train.csv")
df.columns = ["study_id", "L_CC_imid", "R_CC_imid", "L_MLO_imid", "R_MLO_imid","L_CC_birad", "L_CC_den", "L_MLO_bi", "L_MLO_den", "R_CC_birad", "R_CC_den", "R_MLO_bi", "R_MLO_den", "max_bi", "max_den"]
df.info()
# %%
df.to_csv("/home/single4/mammo/mammo/data/updatedata/csv/multiview_train.csv", index = False)
# %%
#bỉrad left max, biraf right max
import pandas as pd 

df = pd.read_csv("/home/single4/mammo/mammo/data/updatedata/csv/multiview_valid.csv")
df["L_birad_max"] = -1
df["L_den_max"] = -1
df["R_birad_max"] = -1
df["R_den_max"] = -1
df.info()
# %%
for i in range(len(df["L_CC_birad"])):
    df.iloc[i,15] = max (df.iloc[i,5], df.iloc[i,7]) #L_birad_max = max(L_birad_CC, L_MLO_bi)
    df.iloc[i,16] = max (df.iloc[i,6], df.iloc[i,8]) #L_den_max
    df.iloc[i,17] = max (df.iloc[i,9], df.iloc[i,11]) #R_bi
    df.iloc[i,18] = max (df.iloc[i,10], df.iloc[i,12]) #R_den

# %%
df.info()
df.to_csv("/home/single4/mammo/mammo/data/updatedata/csv/multiview_valid.csv", index = False)

# %%
