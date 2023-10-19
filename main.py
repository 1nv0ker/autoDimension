import torch
from PIL import Image
import random
import os
import shutil
import imghdr
probabilities = [0.6, 0.2, 0.2] # 样本比例
elements = ['train', 'val', 'test']
IMAGE_PATH='images'
SAVE_PATH='datasets'
CATEGORIES_TXT = 'categories.txt'
RESIZE=608
# Model
#  # local model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')

def createDatasetDir(SAVEINDEX):
    path = os.path.join(SAVE_PATH, 'data'+str(SAVEINDEX))
    if os.path.exists(path)==True:
        SAVEINDEX = SAVEINDEX + 1
        return createDatasetDir(SAVEINDEX)
    else:
        os.mkdir(path)
        return path
def getImgBox(imagePath):
    
    image = Image.open(imagePath)
    size=image.size
    width = size[0]
    height = size[1]
    rate = 0
    targetW = RESIZE
    targetH = RESIZE
    if width>height:
        rate = height/width
        targetH = rate*RESIZE
    else:
        rate = width/height
        targetW = rate* RESIZE
    image = image.resize((int(targetW), int(targetH)))
    newSize=image.size
    newWidth =newSize[0]
    newHeigt = newSize[1]
    results = model(image)
    xyxy = results.pandas().xyxy
    temp = xyxy[0]
    indexList = temp.index.to_list()
    bbox = []
    # columnList = temp.columns.to_list()
    # print(columnList)
    for index in indexList:
        if temp.loc[index, 'name'] == 'bird':
            xmin = temp.loc[index, 'xmin']
            ymin = temp.loc[index, 'ymin']
            xmax = temp.loc[index, 'xmax']
            ymax = temp.loc[index, 'ymax']
            x = (xmax+xmin/2)/newWidth
            y = (ymax+ymin/2)/newHeigt
            w = (xmax-xmin)/newWidth
            h = (ymax-ymin)/newHeigt
            bbox.append([x,y,w,h])
    return [bbox, image]

def main():
    SAVEINDEX=0
    MARKINDEX=0
    fileIndex = 0
    markDatas=[]
    if os.path.exists(SAVE_PATH)==False:
        os.mkdir(SAVE_PATH)
    currentPath = createDatasetDir(SAVEINDEX)
    currentImgPath = currentPath+'/images'
    currentLabelPath = currentPath+'/labels'
    
    if os.path.exists(CATEGORIES_TXT)==True:
        os.remove(CATEGORIES_TXT)
    if os.path.exists(currentImgPath)==False:
        os.mkdir(currentImgPath)
        os.mkdir(currentImgPath+'/train')
        os.mkdir(currentImgPath+'/val')
        os.mkdir(currentImgPath+'/test')
    if os.path.exists(currentLabelPath)==False:
        os.mkdir(currentLabelPath)
        os.mkdir(currentLabelPath+'/train')
        os.mkdir(currentLabelPath+'/val')
        os.mkdir(currentLabelPath+'/test')
    categoriesTxt = open(CATEGORIES_TXT,'w',encoding='utf-8')
    for dirpath in os.listdir(IMAGE_PATH):
        markDatas.append(dirpath)
        path = os.path.join(IMAGE_PATH, dirpath)
        categoriesTxt.write('  '+str(MARKINDEX)+': '+dirpath)
        categoriesTxt.write('\n')
        for filename in os.listdir(path):
            results = random.choices(elements, probabilities)
            tempSavePath = 'train'
            if len(results)>0:
                tempSavePath = results[0]
            txtPath = os.path.join(currentLabelPath, tempSavePath, str(fileIndex)+'.txt')
            tempTxt=open(txtPath,'w')
            sourcePath=os.path.join(path,filename)
            if imghdr.what(sourcePath) == None:
                continue
            bboxes, image = getImgBox(sourcePath)
            if len(bboxes) == 0:
                continue
            for bbox in bboxes:
                x,y,w,h = bbox
                tempTxt.write(str(MARKINDEX)+' '+str(x)+' '+str(y)+' '+str(w)+' '+str(h))
                tempTxt.write('\n')
            tempTxt.close()
            targetPath = os.path.join(currentImgPath, tempSavePath, str(fileIndex)+'.png')
            image.save(targetPath)
            # shutil.copy(sourcePath, targetPath)
            fileIndex = fileIndex + 1
        MARKINDEX = MARKINDEX + 1
    categoriesTxt.close()

main()
