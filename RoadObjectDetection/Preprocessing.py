import json
import numpy as np
def getCenter(x1,y1,x2,y2):
    return (x1+x2)/2,(y1+y2)/2

def getWH(x1,y1,x2,y2):
    return (x2-x1),(y2-y1)
def getVector(vectorOr,vectorNe,indice):
    for i in vectorNe:
        vectorOr[indice]=i
        indice+=1
    return vectorOr
    
et="bike,lane,car,traffic light,traffic sign,person,drivable area,truck,motor,bus,train,rider".split(",")
import ptvsd
# Allow other computers to attach to ptvsd at this IP address and port.
ptvsd.enable_attach(address=('127.0.0.1', 5678), redirect_output=True)

# Pause the program until a remote debugger is attached
ptvsd.wait_for_attach()
#13,13,85(12+5)*5)
#size img 1280x720 px
dx=1280/13
dy=720/13
base = "/home/max/Documents/MiniProjects/bdd100k_labels_release/bdd100k/labels/"
archivo =base+ "bdd100k_labels_images_val.json"
with open(archivo,"r") as file:
    coso = json.loads(file.read())
y_train,y_test=np.zeros(shape=(13,13,85)),np.zeros(shape=(13,13,85))
for c in coso:
    etiquetas=c['labels']
    cont=[[0 for j in range(13)] for i in range(13)]
    for etiq in etiquetas:  
        try:
            e=etiq['box2d']  
        except:
            pass
        xC,yC=np.array(getCenter(e['x1'],e['y1'],e['x2'],e['y2']))
        w,h=np.array(getWH(e['x1'],e['y1'],e['x2'],e['y2']))
        ob=np.array(1)
        dist=np.zeros(12)
        dist[et.index(etiq['category'])]=1
        vector=np.concatenate(([xC,yC],[w,h,ob],dist))
        y_test[int(xC/dx),int(yC/dy)]=getVector(y_test[int(xC/dx),int(yC/dy)],vector,13*cont[int(xC/dx)][int(yC/dy)])
        cont[int(xC/dx)][int(yC/dy)]+=1
import pickle

pickle.dump(y_test,open("pickle_img_val_y_test.pkl","wb"))