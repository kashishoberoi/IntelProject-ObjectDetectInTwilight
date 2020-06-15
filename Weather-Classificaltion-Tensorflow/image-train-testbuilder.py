import glob
from sklearn.model_selection import train_test_split
import os
image_paths = glob.glob("./weatherMultiClassification/*")
train,test = train_test_split(image_paths, test_size=0.3, random_state=42)
class_names = {
        'cloudy':1,
        'rain':2,
        'shine':3,
        'sunrise':4
    }
for class_name in class_names.keys():
    os.mkdir(os.path.join(".\\myweatherdata\\test",class_name))
    os.mkdir(os.path.join(".\\myweatherdata\\train",class_name))
from PIL import Image
for itr in range(len(train)):
    name = train[itr].split(".")[-2].split("\\")[1]
    cls_name = ""
    for class_name in class_names.keys():
        if class_name in name:
            cls_name = class_name
    img = Image.open(train[itr])
    img = img.resize((416,416))
    img = img.convert('RGB')
    img.save(".\\myweatherdata\\train\\"+cls_name+"\\"+name+".jpg")
for itr in range(len(test)):
    name = test[itr].split(".")[-2].split("\\")[1]
    cls_name = ""
    for class_name in class_names.keys():
        if class_name in name:
            cls_name = class_name
    img = Image.open(train[itr])
    img = img.resize((416,416))
    img = img.convert('RGB')
    img.save(".\\myweatherdata\\test\\"+cls_name+"\\"+name+".jpg")