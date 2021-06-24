

import os
import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

os.chdir("/home/single1/BACKUP/SamHUyen/mammo/sam/yolov5/yolov5/")

list_studies = os.listdir("/home/single1/BACKUP/SamHUyen/multi_view_mammo_classification/updatedata/images")
imgs = []
df = pd.read_csv("/home/single1/BACKUP/SamHUyen/mammo/sam/singleview_sam/updatedcsv_singleview.csv")
#txts = []
for study in list_studies:
    study_path = os.path.join("/home/single1/BACKUP/SamHUyen/multi_view_mammo_classification/updatedata/images", study)
    if study not in df["study_id"].values:
        imgs.append(study_path + "/") #study folder 
    #img_list = os.listdir(study_path)
    #for img in img_list:
    #    imgs.append(os.path.join(study_path,img))
    #    txts.append("/media/tungthanhlee/DATA/multi_view_mammo_classification/labels/" + str(img).replace("png","txt"))
print(len(imgs))
for ind in tqdm(range(2190,len(imgs))):
    exe = "python detect.py --weights runs/train/exp11/weights/best.pt --conf 0.25 --device 0 --source " + imgs[ind] + " --save-txt " #+ txts[ind]
    os.system(exe)
