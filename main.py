import torch
from PIL import Image
import os
import shutil
IMAGE_PATH='images'
SAVE_PATH='datasets'
CATEGORIES_TXT = 'categories.txt'
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
    im = imagePath
    results = model(im)
    xyxy = results.pandas().xyxy
    size=Image.open(im).size
    width = size[0]
    height = size[1]
    temp = xyxy[0]
    indexList = temp.index.to_list()
    bbox = []
    # columnList = temp.columns.to_list()
    # print(columnList)
    for index in indexList:
        if temp.loc[index, 'name'] == 'bird':
            box = {}
            xmin = temp.loc[index, 'xmin']
            ymin = temp.loc[index, 'ymin']
            xmax = temp.loc[index, 'xmax']
            ymax = temp.loc[index, 'ymax']
            x = (xmin+xmax/2)/width
            y = (ymin+ymax/2)/height
            w = xmax/width
            h = ymax/height
            bbox.append([x,y,w,h])
    return bbox

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
    if os.path.exists(currentLabelPath)==False:
        os.mkdir(currentLabelPath)
    categoriesTxt = open(CATEGORIES_TXT,'w',encoding='utf-8')
    for dirpath in os.listdir(IMAGE_PATH):
        markDatas.append(dirpath)
        path = os.path.join(IMAGE_PATH, dirpath)
        categoriesTxt.write('  '+str(MARKINDEX)+': '+dirpath)
        categoriesTxt.write('\n')
        for filename in os.listdir(path):
            txtPath = os.path.join(currentLabelPath, str(fileIndex)+'.txt')
            tempTxt=open(txtPath,'w')
            sourcePath=os.path.join(path,filename)
            bboxes = getImgBox(sourcePath)
            for bbox in bboxes:
                x,y,w,h = bbox
                tempTxt.write(str(MARKINDEX)+' '+str(x)+' '+str(y)+' '+str(w)+' '+str(h))
                tempTxt.write('\n')
            tempTxt.close()
            targetPath = os.path.join(currentImgPath, str(fileIndex)+'.jpg')
            shutil.copy(sourcePath, targetPath)
            fileIndex = fileIndex + 1
        MARKINDEX = MARKINDEX + 1
    categoriesTxt.close()

main()
