import json
import numpy as np
def getCenter(x1,y1,x2,y2):
    xA,yA=(x1+x2)/2,(y1+y2)/2
    tdx,tdy=dx*(int(xA/dx)),dy*(int(yA/dy))
    x,y=(xA-tdx)/(dx*tdx+1-tdx),(yA-tdy)/(dy*tdy+1-tdy)
    return xA,yA,x,y

def getWH(x1,y1,x2,y2):
    return (x2-x1)/lx,(y2-y1)/ly

def getVector(vectorOr,vectorNe,indice):
    for i in vectorNe:
        vectorOr[indice]=i
        indice+=1
    return vectorOr
    
et="bike,lane,car,traffic light,traffic sign,person,drivable area,truck,motor,bus,train,rider".split(",")
# import ptvsd
# # Allow other computers to attach to ptvsd at this IP address and port.
# ptvsd.enable_attach(address=('127.0.0.1', 5678), redirect_output=True)

# # Pause the program until a remote debugger is attached
# ptvsd.wait_for_attach()
# #13,13,85(12+5)*5)
# #size img 1280x720 px
gridDiv=13
lx=1280
ly=720
feats=12+5 
n=5
dx=lx/gridDiv
dy=ly/gridDiv

base = "/home/max/Documents/MiniProjects/bdd100k_labels_release/bdd100k/labels/"
archivoTest =base+ "bdd100k_labels_images_val.json"
archivoTrain=base + "bdd100k_labels_images_train.json"
def processing():
	with open(archivoTest,"r") as file:
	    coso = json.loads(file.read())
	y_train,y_test=np.zeros(shape=(gridDiv,gridDiv,85)),np.zeros(shape=(int(len(coso)),gridDiv,gridDiv,feats*n))
	for i,c in enumerate(coso):
	    etiquetas=c['labels']
	    cont=[[0 for j in range(gridDiv)] for i in range(gridDiv)]
	    for etiq in etiquetas:  
	        try:
	            e=etiq['box2d']
	            xC,yC,xr,yr=np.array(getCenter(e['x1'],e['y1'],e['x2'],e['y2']))
	            w,h=np.array(getWH(e['x1'],e['y1'],e['x2'],e['y2']))
	            ob=np.array(1)
	            dist=np.zeros(12)
        	    dist[et.index(etiq['category'])]=1
	            vector=np.concatenate(([xr,yr,w,h,ob],dist))
	            y_test[int(i),int(xC/dx),int(yC/dy)]=getVector(y_test[int(i),int(xC/dx),int(yC/dy)],vector,feats*cont[int(xC/dx)][int(yC/dy)])
	            cont[int(xC/dx)][int(yC/dy)]+=1
	        except:
	            pass
	print("Termine")
	return y_test

y_test=processing()
print("Guardando")

np.savez_compressed("npz_img_val_y_test.npz",y_test=y_test)
# with open("json_img_train_y_train.pkl","w") as outfile:
# 	json.dump(processing(),outfile,ensure_ascii=False))

# data=np.load('arch.npz')
# y_test=data['y_test']