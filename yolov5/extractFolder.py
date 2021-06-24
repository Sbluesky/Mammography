#%%
import pandas as pd
import cv2
import numpy as np
import os
import shutil 
from tqdm import tqdm
#%%
#chuyển data sang 1 folder để dễ dàng thực hiện crop image

stdFl = os.listdir("/home/single4/mammo/mammo/data/vietrad/images")

for std in tqdm(stdFl[3:]):
    stdPath = os.path.join("/home/single4/mammo/mammo/data/vietrad/images",std)
    for img in os.listdir(stdPath):
        imgPath = os.path.join(stdPath, img)
        shutil.copy2(imgPath, "/home/single4/mammo/mammo/sam/yolov5/image/")


# %%
print(len(os.listdir("/home/single4/mammo/mammo/sam/yolov5/image"))) #3516

# %%
#THUẬT TOÁN CROP BẰNG CÁCH DETECT MAMMO
#function to read file
def read_bound_txt(boundbox_path):
    file = open(boundbox_path, 'r')
    boundbox = []
    for line in file:
        boundbox.append(line)
    return boundbox

#Trích xuất bounding box từ fordel detect mammo Yolo5
list_boud_folder = os.listdir("/home/single4/mammo/mammo/sam/yolov5/yolov5/runs/detect/exp/labels")
list_bounding_box = []
for fold in list_boud_folder:
    boundbox_path = os.path.join("/home/single4/mammo/mammo/sam/yolov5/yolov5/runs/detect/exp/labels",fold)
    list_bounding_box.append(boundbox_path + " | " + str(read_bound_txt(boundbox_path)[-1]))

print(len(list_bounding_box))
print(list_bounding_box[:3])
# %%
bound_df = pd.DataFrame()
bound_df["sumary"] = list_bounding_box
bound_df.info()
# %%
bound_df = bound_df["sumary"].str.split(expand=True)
bound_df.head(5)

# %%
image_id = bound_df[0].str.split(pat="/", expand=True)
image_id.head(5)
# %%
#return bounding box format from yolo to normal 
def normalize_boundingbox(x,y,width, height):
  rate = 3072/256
  x_min = x*rate
  x_max = x*rate + width*rate
  y_min = y*rate
  y_max = y*rate + height*rate
  return x_min, x_max, y_min, y_max

bound_box_df = pd.DataFrame()
bound_box_df["image_id"] = image_id[12].str.replace(".txt",".png")
bound_box_df["x_center"] = bound_df[3]
bound_box_df["y_center"] = bound_df[4]
bound_box_df["width"] = bound_df[5]
bound_box_df["height"] = bound_df[6]
bound_box_df.info()
# %%
import matplotlib.pyplot as plt
#check lại id này: 2.16.840.1.113669.632.25.1.32198.20190529095605039.3.png
def comp_image(x_center,y_center,width,height):
    img_width = 2800
    img_height = 3518
    x_max = int((x_center*img_width*2 + width*img_width)/2)
    x_min = int(np.abs(x_max-width*img_width))
    y_max = int((y_center*img_height*2 + height*img_height)/2)
    y_min = int(np.abs(y_max-height*img_height))
    return x_max,x_min,y_max, y_min

def crop_image(img,x_max,x_min,y_max, y_min):
    image = cv2.imread(img,  cv2.IMREAD_GRAYSCALE)
    x1 = max(0,x_min-50)
    x2 = min(image.shape[1], x_max+50)
    y1 = max(0, y_min-50)
    y2 = min(image.shape[0], y_max+50)
    return image[y1:y2, x1:x2 ]

#CUT IMAGES PROCESS
#bound_box_df = pd.read_csv("/home/tungthanhlee/mammo/sam/bound_box_df.csv")
list_imgs = os.listdir("/home/single4/mammo/mammo/sam/yolov5/image")
crop_img_folder = "/home/single4/mammo/mammo/sam/yolov5/cropimages/"
mis_img = [] #images bị mis khi detect
for img in tqdm(list_imgs[152+489+1403+212+189+972:]): 
        if img.find("png") > 0:
            #Check xem img có trong img_id detect không ?
            if img in bound_box_df["image_id"].values: 
                #get x_center,y_center
                
                x_center = float(bound_box_df[bound_box_df["image_id"]==img]["x_center"].values[0])
                y_center = float(bound_box_df[bound_box_df["image_id"]==img]["y_center"].values[0])
                width = float(bound_box_df[bound_box_df["image_id"]==img]["width"].values[0])
                height = float(bound_box_df[bound_box_df["image_id"]==img]["height"].values[0])
                x_max,x_min,y_max, y_min = comp_image(x_center,y_center,width,height)

                #get img
                img_path = os.path.join("/home/single4/mammo/mammo/sam/yolov5/image",img)
                #print(img_path)
                #image = cv2.imread(img_path)
                #crop img
                crop_img = crop_image(img_path, x_max,x_min,y_max, y_min)
                
                #save crop img
                #print(img_path,x_max,x_min,y_max, y_min)
                crop_img_path = os.path.join(crop_img_folder,img )
                #plt.imshow(crop_img)
                #plt.show()
                cv2.imwrite(crop_img_path,crop_img)

            #else:
                #mis_img.append(os.path.join(study_path,img))


#print(num_img)
# %%

#CHECK ẢNH LỖI
img = cv2.imread("/home/single4/mammo/mammo/sam/yolov5/image/2.16.840.1.113669.632.20.20190529.82547741.200230.75.png", cv2.IMREAD_GRAYSCALE)
x_max,x_min,y_max, y_min = 2791, 1979, 3518, 3138
x1 = max(0,x_min-50)
x2 = min(img.shape[1], x_max+50)
y1 = max(0, y_min-50)
y2 = min(img.shape[0], y_max+50)
plt.imshow(img)
#plt.imshow(img[y1:y2, x1:x2 ])
plt.show()

# %%

len(os.listdir("/home/single4/mammo/mammo/sam/yolov5/cropimages"))
# %%
#Bỏ birad 1 và 0 đi
df = pd.read_csv("/home/single4/mammo/mammo/data/vietrad/vietrad.csv")
df = df[(df["birad"] != "BI-RADS 1") & (df["birad"] !="BI-RADS 0")]
df.info()
print(df["birad"].unique())
# %%
#tạo folder studies
studies = df["study_id"].unique()
print("studies: ", len(studies)) #1429

for std in tqdm(studies):
    stdpath = os.path.join("/home/single4/mammo/mammo/data/vietrad/cropimages_vietrad",std)
    os.mkdir(stdpath)
# %%
#chuyển ảnh đã crop sang tùng study
listImg = os.listdir("/home/single4/mammo/mammo/sam/yolov5/cropimages")
for im in tqdm(listImg):
    print(im)
    if im.replace(".png","") in df["image_id"].values:
        imgSour = os.path.join("/home/single4/mammo/mammo/sam/yolov5/cropimages",im)
        std = df[df["image_id"] == im.replace(".png","")]["study_id"].values[0]
        imgDest = os.path.join("/home/single4/mammo/mammo/data/vietrad/cropimages_vietrad",std)
        shutil.move(imgSour,imgDest)
# %%
#check ảnh trong folder + tìm ảnh miss
count = 0
exist_img = []
for std in os.listdir("/home/single4/mammo/mammo/data/vietrad/cropimages_vietrad"):
    stdpath = os.path.join("/home/single4/mammo/mammo/data/vietrad/cropimages_vietrad",std)
    count += len(os.listdir(stdpath))
    for im in os.listdir(stdpath):
        exist_img.append(im.replace(".png",""))
print(count)
for im in df["image_id"].values:
    if im not in exist_img:
        print(im)
#2322, gốc 2326 images => thiếu 4
# %%
