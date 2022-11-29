imena=(['Chang2002.png'])


import numpy as np
import cv2
import math
import pickle
import gc

from tensorflow.keras import models
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD

from tensorflow.keras import backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



    
for imagenamme in imena:
    print(imagenamme)
    image=cv2.imread(imagenamme)
    modelk=models.load_model('model3ll16')
    modelk.summary()



    kk=0
    img=np.zeros((5000,1024),dtype='float16')
    img2=np.empty((0,1024),dtype='float16')
    trr2=np.empty((0),dtype='float')
    kooor2=np.empty((0,3),dtype='float16')
    kooor=np.zeros((5000,3),dtype='float16')
    nnj=0
    iil=15
    ver=0.975
    while iil<33:
        iil=int(iil*1.07)
        Ni,Nj=(iil,iil)
        print(iil/3)
        istep=max(2,int(iil/30))
        jstep=max(2,int(iil/30))
    


        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        #moda=max(1,2*int(iil/20//2)+1)
        #print(moda)
        #gray = cv2.GaussianBlur(gray,(moda,moda),cv2.BORDER_DEFAULT)
        #if iil>32:
        #gray = cv2.GaussianBlur(gray,(3,3),cv2.BORDER_DEFAULT)	
        mean=gray.mean()+00
        
        bordersize=0 # int(iil/3)
        gray=cv2.copyMakeBorder(gray, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[mean] )
        gray=gray/255.0
        #print(imgb.shape[1])
        if Ni<32:
            inter=cv2.INTER_CUBIC 
        else:
            inter=cv2.INTER_AREA






        for i in range(0,gray.shape[0]-Ni-istep,int(istep)):
          #  if kk>6000:
           #     break
            for j in range(0,gray.shape[1]-Nj-jstep,int(jstep)):
                   # print(j)
                #np.reshape(imgb[i:i+Ni,j:j+Nj],361)
		
                img[kk,:]=np.reshape(cv2.resize(gray[i:i+Ni,j:j+Nj],dsize=(32,32),interpolation=inter),1024)
                kooor[kk,:]=((i+iil/2-bordersize),(j+iil/2-bordersize),(iil/3))  #-bordersize -bordersize
                   # print(kooor[kk,:])
                kk=kk+1
                
                if kk>4999:
                    kk=0
                    trr=modelk.predict(img.reshape(img.shape[0],32,32,1),verbose=1,batch_size=500)
                    
                    trr=trr.reshape((trr.shape[0]))
                    img=img[trr>ver]
                    kooor=kooor[trr>ver]
                    img2=np.append(img2,img,axis=0)
                    kooor2=np.append(kooor2,kooor,axis=0)
                    trr2=np.append(trr2,trr[trr>ver],axis=0)
                    img=np.zeros((5000,1024),dtype='float16')

                    kooor=np.zeros((5000,3),dtype='float16')
                
                  #  if kk>6000:
                      #  break

    
    
        nnj=np.append(nnj,kk)
        
        
    print('razbienie')
    print(kk)
    print(imagenamme)
    kooor=kooor[:kk]
    img=img[:kk]

    trr=modelk.predict(img.reshape(img.shape[0],32,32,1),verbose=1,batch_size=500)
    trr=trr.reshape((trr.shape[0]))
    img=img[trr>ver]
    kooor=kooor[trr>ver]
    img2=np.append(img2,img,axis=0)
    kooor2=np.append(kooor2,kooor,axis=0)
    trr2=np.append(trr2,trr[trr>ver],axis=0)
    del img
    del kooor
    del modelk



    print(imagenamme)

    img=np.zeros((5000,1024),dtype='float16')
    kooor=np.zeros((5000,3),dtype='float16')

    modelk=models.load_model('model14ll16')

    ver=0.9997

    kk=0
    nnj=0
    iil=15
    while iil<250:
        iil=int(iil*1.07)
        Ni,Nj=(iil,iil)
        print(iil/1.4)
        istep=max(2,int(iil/30))
        jstep=max(2,int(iil/30))
    


        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        moda=max(3,2*int(iil/10//2)+1)
        print(moda)
        #gray = cv2.GaussianBlur(gray,(moda,moda),cv2.BORDER_DEFAULT)
        #bordersize=int(0.7*iil)
        gray=cv2.copyMakeBorder(gray, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[mean] )
        gray=gray/255.0
        if Ni<32 :
            inter=cv2.INTER_CUBIC 
        else:
            inter=cv2.INTER_AREA
        #print(imgb.shape[1])

        for i in range(0,gray.shape[0]-Ni-istep,int(istep)):
          #  if kk>6000:
           #     break
            for j in range(0,gray.shape[1]-Nj-jstep,int(jstep)):
                   # print(j)
                #np.reshape(imgb[i:i+Ni,j:j+Nj],361)
                img[kk,:]=np.reshape(cv2.resize(gray[i:i+Ni,j:j+Nj],dsize=(32,32),interpolation=cv2.INTER_AREA),1024)
                kooor[kk,:]=((i+iil/2-bordersize),(j+iil/2-bordersize),(iil/1.4))
                   # print(kooor[kk,:])
                kk=kk+1
                
                if kk>4999:
                    kk=0
                    trr=modelk.predict(img.reshape(img.shape[0],32,32,1),verbose=1,batch_size=500)
                    
                    trr=trr.reshape((trr.shape[0]))
                    img=img[trr>ver]
                    kooor=kooor[trr>ver]
                    img2=np.append(img2,img,axis=0)
                    kooor2=np.append(kooor2,kooor,axis=0)
                    trr2=np.append(trr2,trr[trr>ver],axis=0)
                    img=np.zeros((5000,1024),dtype='float16')

                    kooor=np.zeros((5000,3),dtype='float16')
                
                  #  if kk>6000:
                      #  break

    
    
        nnj=np.append(nnj,kk)
        
        
    print('razbienie')
    print(kk)
    print(imagenamme)
    kooor=kooor[:kk]
    img=img[:kk]

    trr=modelk.predict(img.reshape(img.shape[0],32,32,1),verbose=1,batch_size=500)
    trr=trr.reshape((trr.shape[0]))
    img=img[trr>ver]
    kooor=kooor[trr>ver]
    img2=np.append(img2,img,axis=0)
    kooor2=np.append(kooor2,kooor,axis=0)
    trr2=np.append(trr2,trr[trr>ver],axis=0)
    del img
    del kooor




    print(imagenamme)




















    print('vasno')
    print(len(img2))
    
    
    

    trrsave=trr2
    kooorsave=kooor2
    trr=trrsave
    kooor = kooorsave


    

        
    trr=trr.reshape(trr.shape[0],1)
    print(len(trr))
    kooor=np.concatenate((kooor,trr),axis=1)









    iii=0
    koed=5.5
    koef=koed
    njjjjj=math.ceil(int(640*8/koef/4))
    print(njjjjj)
    print(len(kooor))
    while iii<17:
        for i in range(0,kooor.shape[0]-1):

            if (kooor[i,3]<0.5):
                continue
            for j in range(i+1,min(int(i+njjjjj),kooor.shape[0])):
            #for j in range(i+1,kooor.shape[0]):
                if (kooor[j,3]<0.5):
                    continue

                if (kooor[j,3]>0.5) and ((((abs(kooor[i,0] - kooor[j,0]))**2+(abs(kooor[i,1] - kooor[j,1]))**2)**0.5)<(min(kooor[i,2],kooor[j,2])/koef)):
                    kooor[i,0]=(kooor[i,3]*kooor[i,0]+kooor[j,3]*kooor[j,0])/(kooor[j,3]+kooor[i,3])
                    kooor[i,1]=(kooor[i,3]*kooor[i,1]+kooor[j,3]*kooor[j,1])/(kooor[j,3]+kooor[i,3])
                    kooor[i,2]=(kooor[i,3]*kooor[i,2]+kooor[j,3]*kooor[j,2])/(kooor[j,3]+kooor[i,3])

                    kooor[i,3]=(kooor[i,3]+kooor[j,3])/1.0
                    kooor[j,3]=0
            #print(i)

        iii=iii+1  
        koef=koed-iii*0.19
        njjjjj=2*njjjjj
        #if njjjjj>kooor.shape[0]/4:
            #njjjjj=int(kooor.shape[0]/4)

        kooor2=kooor
        print(iii)
        kooor=kooor2[kooor2[:,3]>0.90]

    tf.keras.backend.clear_session()
    gc.collect()



      
    
    filename=imagenamme[:-3]+'pigy'
    pickle.dump((image, kooor), open(filename, 'wb'))
     
    
    
    
    
    
    
 


