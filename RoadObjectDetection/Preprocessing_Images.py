import numpy as np
import cv2 

with open("files_x_train.txt","r") as f:
    lines=f.read().split("\n")
    x_train=np.zeros((len(lines),416,416,3))
    for i,l in enumerate(lines):
        # os.system("ls -lh "+l)
        im=cv2.imread(l,1)
        # print(l)
        x_train[i]=im
        # print(im.shape)
        # input(type(im))

with open("files_x_test.txt","r") as f:
    lines=f.read().split("\n")
    x_test=np.zeros((len(lines),416,416,3))
    for i,l in enumerate(lines):
        # os.system("ls -lh "+l)
        im=cv2.imread(l,1)
        # print(l)
        x_test[i]=im
        # print(im.shape)
        # input(type(im))
np.savez_compressed("npz_x_test.npz",x_test=x_test)        
