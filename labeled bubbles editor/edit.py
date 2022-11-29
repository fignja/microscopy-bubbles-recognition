import numpy as np
import cv2
import math
import pickle
import time
from matplotlib import pyplot as plt
masht=1.7


imagenamme='Image_087.png'    
filename=imagenamme[:-3]+'pigy'


ishod,kooor= pickle.load(open(filename, 'rb'))

vidra=cv2.cvtColor(ishod, cv2.COLOR_RGB2GRAY)
vidra = cv2.equalizeHist(vidra)
vidra = cv2.blur(vidra,(15,15),cv2.BORDER_DEFAULT)
vidra=np.array(vidra)


videlb=False
ishodb=False
addim=False

cv2.namedWindow(imagenamme)


def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode,otris,videlb,jret,iret,dret,verret,iiret,addim,ishod
      
    if event == cv2.EVENT_LBUTTONDOWN:
        ix,iy = (x/masht),(y/masht)
        
    elif event == cv2.EVENT_LBUTTONUP:
        otris=ishod.copy()
        x,y = (x/masht),(y/masht)
        for i,j,d,ver in (kooor):
                otris=cv2.circle(otris, (int(j),int(i)), int(d/2),(255,0,0),1)
            
        if (int(math.sqrt((ix-x)**2  +(iy-y)**2))>4):
            addim=True
            videlb=False
      #  if mode == True:
            otris=cv2.circle(otris, (int((ix+x)/2),int((iy+y)/2)), int(math.sqrt((ix-x)**2  +(iy-y)**2)/2),(0,255,0),1)
            jret=int((ix+x)/2)
            iret=int((iy+y)/2)
            dret=int(math.sqrt((ix-x)**2  +(iy-y)**2))
            verret=100
        

        
            cv2.imshow(imagenamme,otris)
        
            
    elif event == cv2.EVENT_RBUTTONDOWN:
        otris=ishod.copy()
        for i,j,d,ver in (kooor):
                otris=cv2.circle(otris, (int(j),int(i)), int(d/2),(255,0,0),1)
        
        
        videlb=True
        addim=False
        ix,iy = (x/masht),(y/masht)
        
        for iim in range(0,kooor.shape[0]):
            #for j in range(i+1,kooor.shape[0]):

               # i,j,d,ver
                if abs(  ((kooor[iim,0]-iy)**2+(kooor[iim,1]-ix)**2)**0.5   -kooor[iim,2]/2)  < 5:
                    iiret=iim
                    
                    otris=cv2.circle(otris, (int(kooor[iim,1]),int(kooor[iim,0])), int(kooor[iim,2]/2),(0,0,255),1)
        
                                        
                    break
                    
        
        
        
        
        
        
        
        
        

cv2.setMouseCallback(imagenamme,draw_circle)

while(1):
    if ishodb:
         
        #print(int(ishod.shape[1]*masht),int(ishod.shape[0]*masht))
        #print()
        #print()


        cv2.imshow(imagenamme,cv2.resize(ishod,dsize=((  int(ishod.shape[1]*masht),int(ishod.shape[0]*masht))     )    ))
    else:    
        if videlb or addim:
            cv2.imshow(imagenamme,cv2.resize(otris,dsize=((  int(ishod.shape[1]*masht),int(ishod.shape[0]*masht))     )    ))


        
        else:
            otris=ishod.copy()
            for i,j,d,ver in (kooor):
                otris=cv2.circle(otris, (int(j),int(i)), int(d/2),(255,0,0),1)
            cv2.imshow(imagenamme,cv2.resize(otris,dsize=((  int(ishod.shape[1]*masht),int(ishod.shape[0]*masht))     )    ))

    
    k = cv2.waitKey(1) & 0xFF
    
    if k == ord(' '):
        ishodb=not ishodb
        videlb=False
        addim=False
    if k == ord('b'):
        if addim:
            kooor = np.concatenate((kooor,np.array([[iret,jret,dret,verret]])))
            addim=False            
            
            a=int(jret-0.65*dret)
            b=int(jret+0.65*dret)
            c=int(iret-0.65*dret)
            d=int(iret+0.65*dret)
            
            print(a,b,c,d)
            nnado=True
            if dret<8:
                nnado=False
            if a<0:

                nnado=False
                a=0

            if a>640:
                nnado=False
                a=640
            if b<0:
                nnado=False
                b=0
            if b>640:
                nnado=False
                b=640    
            if c<0:
                nnado=False
                c=0
            if c>480:
                nnado=False
                c=480    
            if d<0:
                nnado=False
                d=0
            if d>480:
                nnado=False
                d=480        

            if nnado:
                
                imna='./da/'+str(time.time())+'.jpg'
            
                cv2.imwrite(imna, cv2.resize(vidra[c:d,a:b],dsize=(19,19),interpolation=cv2.INTER_AREA))
            
            
            
            

        if videlb:
            
            
            
            iret,jret,dret,verret = kooor[iiret] 
            
            videlb=False
            
            a=int(jret-0.65*dret)
            b=int(jret+0.65*dret)
            c=int(iret-0.65*dret)
            d=int(iret+0.65*dret)
            
            print(a,b,c,d)
            nnado=True
            if dret<8:
                nnado=False
            if a<0:

                nnado=False
                a=0

            if a>640:
                nnado=False
                a=640
            if b<0:
                nnado=False
                b=0
            if b>640:
                nnado=False
                b=640    
            if c<0:
                nnado=False
                c=0
            if c>480:
                nnado=False
                c=480    
            if d<0:
                nnado=False
                d=0
            if d>480:
                nnado=False
                d=480        

            if nnado:
                
                imna='./ne/'+str(time.time())+'.jpg'
            
                cv2.imwrite(imna, cv2.resize(vidra[c:d,a:b],dsize=(19,19),interpolation=cv2.INTER_AREA))
            
                        
            
            kooor=np.delete(kooor,iiret,axis=0)
            videlb=False
                
        
        
    if k == ord('t'):
       # gray5=cv2.imread('Image_001.tif')
        mode = not mode        
    elif k == 27:
        break    

cv2.destroyAllWindows()


pickle.dump((ishod, kooor), open(filename, 'wb'))
gray4=ishod
for i,j,d,ver in (kooor):
    gray4=cv2.circle(gray4, (int(j),int(i)), int(d/2),(255,255,255),1)

kor=kooor[:,2]*1.64
cv2.imwrite(imagenamme[:-4]+'_sredd = '+str("{:7.3f}".format(np.mean(kor)))+'.jpg', gray4)

plt.hist(kor,bins=15)

plt.savefig(imagenamme[:-4]+'_raspred'+'.jpg')

print(imagenamme)
print("{:7.3f}".format(np.mean(kor)))
#np.savetxt(imagenamme[:-3]+'txt', kor, fmt='%1.1f')
kooor[:,0]=480-kooor[:,0]
np.savetxt(imagenamme[:-4]+'.txt', kooor[:,:3], fmt='%1.1f')

