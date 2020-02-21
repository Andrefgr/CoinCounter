import cv2
import numpy as np
from matplotlib import pyplot as plt
import bwLabel
import psColor 
import math

name = 'assets/12.jpg'
img = cv2.imread(name)
img_copy = img.copy()


coin=[[[14000,16000],2],
      [[11000,12000],1],
      [[12001,14000],0.5],
     [[9400,11000],0.2],
      [[6700,7500],0.1],
      [[7500,9400],0.05],
      [[5000,6700],0.02],
      [[3000,5000],0.01]]

coinname=["2 euros","1 euro","50 centimos", "20 centimos", "10 centimos","5 centimos", "2 centimos","1 centimo"]



def des_moeda(value):
        d=list(dicionario.values())
        k=list(dicionario.keys());
        
        for b in range(len(d)):
            if(value>=d[b][0][0] and value<=d[b][0][1]):
                return k[b],d[b][1]

dicionario= dict(zip(coinname,coin))

cv2.imshow("image",img)
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_one = img[:, :, 2]

blur = cv2.GaussianBlur(img_one,(5,5),0)
#blur = cv2.GaussianBlur(gray,(5,5),0)
ret, th = cv2.threshold(blur,0,255,cv2.THRESH_OTSU)

cv2.imshow("after threshold",th)

kernel = np.ones((3,3),np.uint8)
b1=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

op = cv2.morphologyEx(th, cv2.MORPH_CLOSE, b1,iterations=3)
cv2.imshow("after op",op)

erode = cv2.erode(op,kernel,iterations=25)
cv2.imshow("after erode",erode)

b1=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(22,22))
dilate = cv2.dilate(erode,b1,iterations=1)

cv2.imshow("after dilate",dilate)
#final = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
contourSeq,grayscale,out2 = bwLabel1.labeling(dilate) 
colorMap = psColor2.CreateColorMap(grayscale)
colorImage = psColor2.Gray2PseudoColor(out2, colorMap);
cv2.imshow("after labelling", colorImage)
total = 0

for cnt in contourSeq:
      
      
      M = cv2.moments(cnt)
      cx = int(M['m10']/M['m00'])
      cy = int(M['m01']/M['m00'])
      (x,y),radius = cv2.minEnclosingCircle(cnt)
      center = (int(x),int(y))
      area = cv2.contourArea(cnt)  
      perimeter = cv2.arcLength(cnt,True)
      if area < 3500. or area > 16000.:
         continue
    
      (x,y),radius = cv2.minEnclosingCircle(cnt)
      center = (int(x),int(y))
      radius = int(radius)
    
      second_area=np.pi*radius**2
      if(abs(area-second_area)>7000):
         continue
     
      areacm =  area * 2.54 / 72
      print()
      
      cv2.circle(img_copy,center,radius,(0,0,255),2)
      cv2.putText(img_copy,str(des_moeda(area)[0]),(cnt[0][0][0],cnt[0][0][1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2)
      total+= float(des_moeda(area)[1])
      print(des_moeda(area)[0], " area = ", area)

cv2.putText(img_copy,str("valor total : " + str(round(total, 3)) + " euros"),(10,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2) 
cv2.imshow("Image " + name, img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()