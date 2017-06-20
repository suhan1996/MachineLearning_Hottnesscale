#!/usr/bin/env python
#coding=utf-8
import os
from PIL import Image, ImageDraw
import cv,cv2

def detect_object(image):
    '''检测图片,获取人脸在图片中的坐标'''
    grayscale = cv.CreateImage((image.width, image.height), 8, 1)
    cv.CvtColor(image, grayscale, cv.CV_BGR2GRAY)

    cascade = cv.Load("haarcascade_frontalface_alt_tree.xml")
    rect = cv.HaarDetectObjects(grayscale, cascade, cv.CreateMemStorage(), 1.1, 2,
        cv.CV_HAAR_DO_CANNY_PRUNING, (20,20))

    result = []
    for r in rect:
        result.append((r[0][0], r[0][1], r[0][0]+r[0][2], r[0][1]+r[0][3]))
    #print result
    return result
   
def processFace(infile,outfile):
    '''在原图上框出头像并且截取每个头像到单独文件夹'''
    image = cv.LoadImage(infile);
    img = cv2.imread(infile)

    if image:
        faces = detect_object(image)

    im = Image.open(infile)
    path = os.path.abspath(infile)
    save_path = os.path.splitext(path)[0]+""
    try:
        #os.mkdir(save_path) put it into specific dir
        pass
    except:
        pass
    sizeList = []
    if faces:
        draw = ImageDraw.Draw(im)
        count = 0
        for f in faces:
            count += 1
            size = (float(f[2]-f[0]) ** 2)/img.shape[0]/img.shape[1]
            center = ((f[3]+f[1])/2./img.shape[0],(f[2]+f[0])/2./img.shape[1])                       ##print the size and location of the face
            sizeList.append(size)
            #print f
            #print img.shape
            #print "face size: " +str(size)
            #print "face center: "+str(center)
            draw.rectangle(f, outline=(255, 0, 0))
            a = im.crop(f)
            file_name = os.path.join("output",outfile+".jpg") #all to output
            #print file_name
            a.save(file_name)

        #drow_save_path = os.path.join(save_path,"out.jpg") draw rectangle on original face
        #im.save(drow_save_path, "JPEG", quality=80)
    else:
        print "Error: cannot detect faces on %s" % infile

if __name__ == "__main__":
    for i in range(1,3105):
        try:
            processFace(str(i) + ".jpg",str(i))
        except:
            pass
