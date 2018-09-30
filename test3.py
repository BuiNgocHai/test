import json
import numpy as np
import cv2
import os
import random

def rotateImage(image, angel):
    height = image.shape[0]
    width = image.shape[1]
    height_big = height *2
    width_big = width *2
    image_big = cv2.resize(image, (width_big, height_big))
    image_center = (width_big/2, height_big/2)
    x = round(random.uniform(0.75,0.8),2)
    #print(x)
    rot_mat = cv2.getRotationMatrix2D(image_center,angel, x)
    result = cv2.warpAffine(image_big, rot_mat, (width_big, height_big), flags=cv2.INTER_LINEAR,borderValue =255)
    return result

#imageOriginal = cv2.imread("0_0.png",1)
#imageOriginal = cv2.resize(imageOriginal, (60,80))
#imageRotated= rotateImage(imageOriginal, 3)
def build(s,s2):
    img = cv2.imread(s,1)
    from random import randint
    x =randint(-7,7)
    img = rotateImage(img,x)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j,0] >= 252:
                img[i,j] = 255
    cv2.imwrite(s2,img)
def show(s,n,l,s2):
    if n == 1:
        n1 = str(n)
        s = s[:-5]+n1+s[-4:]
    #print(s)
    if l == 1:
        from random import randint
        a = randint(0,35)
        a = str(a)
        s = s[:-5] + a + s[-4:]
        #print(s)
    img1 = cv2.imread(s2, 1)
    img2 = cv2.imread(s, 1)
    #img3 = cv2.imread('white.png',1)
    from random import randint
    x=randint(5,20)
    wc=randint(50,150)
    hc=randint(50,150)
    d = randint(-7,7)
    #print(x)
    #print(img1.shape)
    if l==0:
        img2 = rotateImage(img2,d)
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            if img2[i,j,0] >= 252:
                img2[i,j] = 255

    h2, w2 = img2.shape[:2]
    img2 = cv2.resize(img2,(w2,h2))
    #img3= cv2.resize(img3,(x,50))
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    #h3, w3 = img3.shape[:2]
    vis = np.zeros((max(h1,h2),w1+w2,3),np.uint8)
    #vis = np.zeros((max(h1, max(h3,h2)), w1+w2+w3,3), np.uint8)
    vis[:h1, :w1,:3] = img1
    #vis[:h3, w1:w1+w3,:3] = img3
   # vis[:h2, w1+w3:w1+w3+w2,:3] = img2
    #print(h1,h2)
    if l == 1:
        from random import randint
        p = randint(w1,w2+w1)
        vis [(int)(h1/2):(int)(h1/2)+h2, w1-20:w1+w2-20] = img2
    else:
        vis[:h2,w1:w1+w2] =img2
    #cv2.imshow('abc',vis)
   # cv2.waitKey(500)
    cv2.imwrite(s2,vis)
def savve(s2):
    img1 = cv2.imread(s2,1)
    img2 = cv2.imread('white.png',1)
    h1,w1 = img2.shape[:2]
    rows,cols = img1.shape[:2]
    img2 = cv2.resize(img2,(h1+rows,w1+cols))
    cv2.imwrite('white.png',1)
    img2 = cv2.imread('white.png',1)
    roi = img2[0:rows, 0:cols]
    img1gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img1gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img2_fg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    img1_bg = cv2.bitwise_and(img1,img1,mask = mask)
    dst = cv2.add(img1_bg,img2_fg)
    img2[0:rows, 0:cols ] = dst #90 90
    img2 = img2[0:rows,0:cols] #80 110
    #os.chdir('/home/vicker/Desktop/work/new')
    os.chdir('/home/vicker/Desktop/work/new')
    cv2.imwrite(s2,img2)
def background(s2):
    n = random.randint(15,76)
    st = str(n)+'.jpg'
    os.chdir('/home/vicker/Desktop/work/new')
    img1 = cv2.imread(s2,1)
    os.chdir('/home/vicker/Desktop/work/background')
    img2 = cv2.imread(st,1)
    print(st)
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    if h2<h1 or w2<w1:
        img2 = cv2.resize(img2,(w1,h1))

    h2,w2 = img2.shape[:2]
    x = random.randint(0,h2-h1)
    y = random.randint(0,w2-w1)
    img2 = img2[x:h1+x,y:w1+y]
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    rows,cols = thresh.shape
    for i in range(thresh.shape[0]):
        for j in range(thresh.shape[1]):
            if thresh[i,j] == 255:
                thresh[i,j] = 0
                img2[i,j] = thresh[i,j]
    os.chdir('/home/vicker/Desktop/work/new')
    cv2.imwrite(s2,img2)    
def savve1(s2):
    img = cv2.imread(s2,1)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j,0] == 0:
                img[i,j] = 255
    os.chdir('/home/vicker/Desktop/work/new')
    cv2.imwrite(s2,img)
with open('lables.json') as f:
    data = json.load(f)
#print(data.get('nham'))
f = open('address.txt','r')
s = f.readlines()
words = [x.rstrip() for x in s]
poz = int(0)
for i in words:
    os.chdir('/home/vicker/Desktop/work/0/dist')
    i = i.replace(',',' ,')
    i = i.replace('.',' .')
    k = i.split()
    s1 = '1'
    z=0
    poz1 = str(poz)
    s2 = 'test_0.png'
    s2 = s2[:-5] +poz1+s2[-4:]
    print(s2)
    print(k)
    for j in range(len(k)):
        l = 0 
        if k[j] == ',' or k[j] == '.':
            l=1
        for key,value in data.items():
            t=0
            if k[j] == value:
                s1 = k[j]
                if(s1[0].isupper() == True and k[j].isupper() == False ):
                    t=1
               # print(key)
                if z==0:
                   build(key,s2)
                   z=1
                elif z==1:
                    show(key,t,l,s2)
                break
    poz+=1
    savve1(s2)
    background(s2)
    
