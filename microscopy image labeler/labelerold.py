from imutils import paths
imena=list(paths.list_images('imgs'))


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

ms=10000
bs=500

modelk=models.load_model('kerad20.h5')
modelk.summary()
    
for imagenamme in imena:
    print(imagenamme)
    image=cv2.imread(imagenamme)





    kk=0
    img=np.zeros((ms,361),dtype='float16')
    img2=np.empty((0,361),dtype='float16')
    trr2=np.empty((0),dtype='float')
    kooor2=np.empty((0,3),dtype='float16')
    kooor=np.zeros((ms,3),dtype='float16')
    nnj=0
    iil=9
    ver=0.995
    while iil<300:
        iil=int(iil*1.13)
        patch_size=(iil,iil)
        #patch_size=(19,19)
        istep=max(2,int(iil/50))
        jstep=max(2,int(iil/50))
        scal=1  #9/iil
        #imgb=gray.copy()
        #imgb=cv2.resize(imgb,dsize=(int(gray.shape[1]*scal),int(gray.shape[0]*scal)))

        Ni,Nj = patch_size

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        #gray = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_DEFAULT)
        cv2.blur(gray,(5,5),cv2.BORDER_DEFAULT)
        mean=gray.mean()+30
        
        bordersize=int(0.085*iil+2)
        gray=cv2.copyMakeBorder(gray, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[mean] )
        gray=gray/255.0
        
        
        
        
        
        
        
        
        #print(imgb.shape[1])

        for i in range(0,gray.shape[0]-Ni-istep,int(istep)):
          #  if kk>6000:
           #     break
            for j in range(0,gray.shape[1]-Nj-jstep,int(jstep)):
                   # print(j)
                #np.reshape(imgb[i:i+Ni,j:j+Nj],361)
                img[kk,:]=np.reshape(cv2.resize(gray[i:i+Ni,j:j+Nj],dsize=(19,19),interpolation=cv2.INTER_AREA),361)
                kooor[kk,:]=((i/scal+iil/2-bordersize),(j/scal+iil/2-bordersize),(iil/1.3))
                   # print(kooor[kk,:])
                kk=kk+1
                
                if kk>ms-1:
                    kk=0
                    trr=modelk.predict(img.reshape(img.shape[0],19,19,1),verbose=1,batch_size=bs)
                    
                    trr=trr.reshape((trr.shape[0]))
                    img=img[trr>ver]
                    kooor=kooor[trr>ver]
                    img2=np.append(img2,img,axis=0)
                    kooor2=np.append(kooor2,kooor,axis=0)
                    trr2=np.append(trr2,trr[trr>ver],axis=0)
                    img=np.zeros((ms,361),dtype='float16')

                    kooor=np.zeros((ms,3),dtype='float16')
                
                  #  if kk>6000:
                      #  break

    
    
        nnj=np.append(nnj,kk)
        
        
    print('razbienie')
    print(kk)
    print(imagenamme)
    kooor=kooor[:kk]
    img=img[:kk]
    trr=modelk.predict(img.reshape(img.shape[0],19,19,1),verbose=1,batch_size=bs)
                    
    trr=trr.reshape((trr.shape[0]))
    img=img[trr>ver]
    kooor=kooor[trr>ver]
    img2=np.append(img2,img,axis=0)
    kooor2=np.append(kooor2,kooor,axis=0)
    trr2=np.append(trr2,trr[trr>ver],axis=0)
    del img
    del kooor
    
    #trr=modelk.predict_proba(img)[:,1]

    print(imagenamme)

    
    print('vasno')
    print(len(img2))

    trr=trr.reshape(trr.shape[0],1)
    print(len(trr))
    kooor=np.concatenate((kooor,trr),axis=1)




    ##gray3=cv2.imread(imagenamme)
    ###gray3 = cv2.cvtColor(gray3, cv2.COLOR_RGB2GRAY)
    ##gray3 = cv2.equalizeHist(gray3)
    #for i,j,d,vv in (kooor):
       # gray3=cv2.rectangle(image, (int(j-d/2),int(i-d/2)), (int(j+d/2),int(i+d/2)), int(255*vv))


    #cv2.imwrite(imagenamme[:-4]+'aa52'+'.jpg', gray3)



    iii=0
    koed=5.5
    koef=koed
    njjjjj=math.ceil(int(640*8/koef/4))
    print(njjjjj)
    print(len(kooor))
    while iii<18:
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
     
    
    
    
    
    
    
 


