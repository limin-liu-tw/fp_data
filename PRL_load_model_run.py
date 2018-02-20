import cv2
import MyLib as MyL
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

ShowAll = False
BGloop= 10
PixNo = 25
PixNo_1_2 = 12
DirNo = 15
DirAgl= [i*(180//DirNo) for i in range(DirNo)]
RowNo = lambda row: row//PixNo
ColNo = lambda col: col//PixNo
WW = 25
FW = 12
FeaNo = 21

if __name__ == '__main__':
 if len(sys.argv) != 3: 
  print('Usage: {} image_name'.format(sys.argv[0]))
  print('Example: python3 PRL_load_model.py Model_Name For_PRL_a002_03.png')  
  # python3 PRL_load_model_run.py test For_PRL_a002_08.png
  sys.exit(0)
 
 #
 # 1. load image (after 3x3 blur image)
 #
 print("1. Loading image... "+sys.argv[2])
 img = cv2.imread(sys.argv[2],0)  
 fm_base=np.asarray(img)
 fm=fm_base.copy()

 #
 # 2. display the original image
 #
 print("2. display image... ")
 plt.title(sys.argv[2][-11:])
 plt.imshow(fm, cmap=plt.cm.gray)
 ImgPath='D:\\Fingerprint\\paper8_NN\\P8NN_Images\\'+sys.argv[1]+'_'+sys.argv[2][-11:-4]+"_1.png"
 plt.savefig(ImgPath, dpi=600, bbox_inches='tight')
 if ShowAll: plt.show()

 #
 # 3. find foreground information
 #
 print("3. find foreground info... ")
 fpfg=MyL.FP_FindBG(plt, fm, False)

 #
 # 4. load model
 #
 print("4. Loading model/weights...")
 from keras.models import model_from_yaml
 MdlSourPath='D:\\Fingerprint\\paper8_NN\\P8NN_Model\\'
 mdl_fn=MdlSourPath+sys.argv[1]+".yaml"
 wgt_fn=MdlSourPath+sys.argv[1]+".h5"
 yaml_file = open(mdl_fn)
 loaded_model_yaml = yaml_file.read()
 yaml_file.close()
 loaded_model = model_from_yaml(loaded_model_yaml)
 loaded_model.load_weights(wgt_fn)

 #
 # 5. prepare 21-feature
 # 
 print("5. prepare 21-feature...")
 height, width = fm.shape[0], fm.shape[1] 
 NoHeight, NoWidth = RowNo(height), ColNo(width)

 X_data = np.full((NoHeight*NoWidth, FeaNo), 0, dtype=np.double)
 X_data_idx=np.full((NoHeight*NoWidth, 2), 0, dtype=int)
 fpdir = np.full((NoHeight,NoWidth), -1, dtype=float)
 fpdir_prob = np.full((NoHeight,NoWidth), -1, dtype=float)
 fpdir_porg = np.full((NoHeight,NoWidth), -1, dtype=float)
 k = -1
 for i in range(NoHeight):
  for j in range(NoWidth):
   if fpfg[i][j]:
    a,b,c,d=i*PixNo,(i+1)*PixNo,j*PixNo,(j+1)*PixNo
    img, k = fm[a:b,c:d], k+1  # img is a 25x25 blocks 
    X_data_idx[k,0], X_data_idx[k,1] = i, j
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    Vx, Vy = np.sum(2*sobelx*sobely), np.sum(sobelx**2-sobely**2)
    X_data[k,0], X_data[k,1], X_data[k,2] =  Vx, Vy, Vy/Vx   
    Reg1x, Reg1y= np.sum(2*sobelx[:,6:19]*sobely[:,6:19]), np.sum(sobelx[:,6:19]**2-sobely[:,6:19]**2)
    X_data[k,3], X_data[k,4], X_data[k,5] =  Reg1x, Reg1y, Reg1y/Reg1x 
    Reg2x, Reg2y= np.sum(2*sobelx[6:19,:]*sobely[6:19,:]), np.sum(sobelx[6:19,:]**2-sobely[6:19,:]**2)
    X_data[k,6], X_data[k,7], X_data[k,8] =  Reg2x, Reg2y, Reg2y/Reg2x 
    Reg3x, Reg3y= np.sum(2*sobelx[0:13,  0:13]*sobely[0:13,0:13]), np.sum(sobelx[0:13,0:13]**2-sobely[0:13,0:13]**2)
    X_data[k,9], X_data[k,10], X_data[k,11] =  Reg3x, Reg3y, Reg3y/Reg3x 
    Reg4x, Reg4y= np.sum(2*sobelx[0:13, 13:25]*sobely[0:13,13:25]), np.sum(sobelx[0:13,13:25]**2-sobely[0:13,13:25]**2)
    X_data[k,12], X_data[k,13], X_data[k,14] =  Reg4x, Reg4y, Reg4y/Reg4x 
    Reg5x, Reg5y= np.sum(2*sobelx[13:25, 0:13]*sobely[13:25,0:13]), np.sum(sobelx[13:25,0:13]**2-sobely[13:25,0:13]**2)
    X_data[k,15], X_data[k,16], X_data[k,17] =  Reg5x, Reg5y, Reg5y/Reg5x     
    Reg6x, Reg6y= np.sum(2*sobelx[13:25,13:25]*sobely[13:25,13:25]), np.sum(sobelx[13:25,13:25]**2-sobely[13:25,13:25]**2)    
    X_data[k,18], X_data[k,19], X_data[k,20] =  Reg6x, Reg6y, Reg6y/Reg6x 
   
 #
 # 6. normalize and make prediction
 #
 print("6. normalize and make prediction...") 
 import sklearn.preprocessing as preprocessing
 X_data_normalize = preprocessing.normalize(X_data)
 prediction = loaded_model.predict_classes(X_data_normalize)

 #===========
 weightTBL = MyL.prepWeight()
 predict_proba = pd.DataFrame.from_records(loaded_model.predict_proba(X_data_normalize))
 prob_arr  = predict_proba.as_matrix()
 prob_list = predict_proba.max(axis=1)
 #print("Prob 1: ",predict_proba.shape,type(predict_proba),"predict_proba=====")#,predict_proba)
 #print("Prob 2: ",prob_list.shape,"prob_list=====")
 #print("Prob 3: ",weightTBL.shape,"weightTBL=====")
 #print("Prob 4: ",prediction.shape,"prediction=====")
 #print("Prob 5: ",prob_arr.shape,"prob_arr=====")

 for i in range(prediction.shape[0]):
    fpdir[X_data_idx[i,0]][X_data_idx[i,1]] = prediction[i]
    fpdir_porg[X_data_idx[i,0]][X_data_idx[i,1]]  = prob_list[i]
    fpdir_prob[X_data_idx[i,0]][X_data_idx[i,1]]  = sum(weightTBL[prediction[i]]*prob_arr[i])
    #if i < 3:
    #  print(i, prediction[i], prob_list[i], sum(weightTBL[prediction[i]]*prob_arr[i]))
    #  for j in range(180):
    #     print('{:2d} {:.3f} {:.3f}'.format(j, weightTBL[prediction[i],j], prob_arr[i,j]))
    #  print(weightTBL[prediction[i],prediction[i]-10:prediction[i]+10], prob_list[i][prediction[i]-10:prediction[i]+10])


 #
 # 7. write line seg. on fingerprint 
 #
 print("7. write line seg. on fingerprint...") 

 fmdir=MyL.UT_SetLine2(plt, fpfg, fpdir, fm.shape, False, 'black',3)

 plt.title(sys.argv[2][-11:])
 plt.imshow(fmdir, cmap=plt.cm.gray)
 ImgPath='D:\\Fingerprint\\paper8_NN\\P8NN_Images\\'+sys.argv[1]+'_'+sys.argv[2][-11:-4]+"_2.png"
 plt.savefig(ImgPath, dpi=600, bbox_inches='tight')
 if ShowAll: plt.show()
 
 #
 # 8. write line seg.  
 #
 print("8. write line seg. ...") 
 axs=[[None for _ in range(NoWidth)]]*NoWidth
 MyL.UT_SetLine(plt, fpfg, fpdir, fm, axs, False, 'white',3)
 plt.title(sys.argv[2][-11:])
 plt.imshow(fm, cmap=plt.cm.gray)
 ImgPath='D:\\Fingerprint\\paper8_NN\\P8NN_Images\\'+sys.argv[1]+'_'+sys.argv[2][-11:-4]+"_3.png"
 plt.savefig(ImgPath, dpi=600, bbox_inches='tight')
 if ShowAll: plt.show()

 #
 # 9. tri-color image
 #
 print("9. Tri-color. image...") 
 fmt=np.full(fm.shape, 192, dtype=int)
 fmred=MyL.UT_SetTri(plt, fpfg, fpdir, fmt, 2)
 plt.title(sys.argv[2][-11:])
 plt.imshow(fmt, cmap=plt.cm.gray)
 ImgPath='D:\\Fingerprint\\paper8_NN\\P8NN_Images\\'+sys.argv[1]+'_'+sys.argv[2][-11:-4]+"_5.png"
 plt.savefig(ImgPath, dpi=600, bbox_inches='tight')
 #plt.show()
 plt.title(sys.argv[2][-11:])
 plt.imshow(fmred, cmap=plt.cm.gray)
 ImgPath='D:\\Fingerprint\\paper8_NN\\P8NN_Images\\'+sys.argv[1]+'_'+sys.argv[2][-11:-4]+"_6.png"
 plt.savefig(ImgPath, dpi=600, bbox_inches='tight')
 #plt.show()

 #
 # 10. prob image
 #
 print("10. Prob. image...") 
 fmprb=np.full(fm.shape, 192, dtype=int)
 MyL.UT_SetGray(plt, fpfg, fmprb, fpdir_prob)
 plt.title(sys.argv[2][-11:])
 plt.imshow(fmprb, cmap=plt.cm.gray)
 ImgPath='D:\\Fingerprint\\paper8_NN\\P8NN_Images\\'+sys.argv[1]+'_'+sys.argv[2][-11:-4]+"_4.png"
 plt.savefig(ImgPath, dpi=600, bbox_inches='tight')
 #plt.show()

 #
 # 11. pixel three color image
 #
 print("11. Tri-color image...") 
 print("11.a. prepare 21-feature...")
 height, width = fm.shape[0], fm.shape[1] 
 NoHeight, NoWidth = height-PixNo+1, width-PixNo+1
 print(NoHeight, NoWidth) 
 #X_pix_data = np.full((NoHeight*NoWidth, FeaNo), 0, dtype=np.double)
 X_pix_data = np.full((NoWidth, FeaNo), 0, dtype=np.double)
 X_pix_idx=np.full((NoHeight*NoWidth, 2), 0, dtype=int)
 pix_dir = np.full((NoHeight,NoWidth), -1, dtype=float)      # pixel direction
 pix_dir_prob = np.full((NoHeight,NoWidth), -1, dtype=float) # pixel dir probabolity
 pix_consensus = np.full((NoHeight,NoWidth), -1, dtype=float) # consensus value
 print("pix_dir.shape, pix_dir_prob.shape: ", pix_dir.shape, pix_dir_prob.shape)
 print("X_pix_data.shape, X_pix_idx.shape: ", X_pix_data.shape, X_pix_idx.shape, "\n processing i: ", end='')
 
 for i in range(NoHeight):
  # need to do this by parts to prevent out-of-memory problem
  k = -1
  #print('.', end='') 
  print('{:4d}'.format(i), end='') 
  sys.stdout.flush()
  for j in range(NoWidth):
   a,b,c,d=i,i+25,j,j+25
   img, k = fm[a:b,c:d], k+1  # img is a 25x25 blocks
   #print(i,j, a, b, c, d)
   X_pix_idx[k,0], X_pix_idx[k,1] = i+12, j+12

   sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
   sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
   Vx, Vy = np.sum(2*sobelx*sobely)+sys.float_info.epsilon, np.sum(sobelx**2-sobely**2)
   X_pix_data[k,0], X_pix_data[k,1], X_pix_data[k,2] =  Vx, Vy, Vy/Vx
   
   Reg1x, Reg1y= np.sum(2*sobelx[:,6:19]*sobely[:,6:19])+sys.float_info.epsilon, np.sum(sobelx[:,6:19]**2-sobely[:,6:19]**2)
   X_pix_data[k,3], X_pix_data[k,4], X_pix_data[k,5] =  Reg1x, Reg1y, Reg1y/Reg1x 

   Reg2x, Reg2y= np.sum(2*sobelx[6:19,:]*sobely[6:19,:])+sys.float_info.epsilon, np.sum(sobelx[6:19,:]**2-sobely[6:19,:]**2)
   X_pix_data[k,6], X_pix_data[k,7], X_pix_data[k,8] =  Reg2x, Reg2y, Reg2y/Reg2x 

   Reg3x, Reg3y= np.sum(2*sobelx[0:13,  0:13]*sobely[0:13,0:13])+sys.float_info.epsilon, np.sum(sobelx[0:13,0:13]**2-sobely[0:13,0:13]**2)
   X_pix_data[k,9], X_pix_data[k,10], X_pix_data[k,11] =  Reg3x, Reg3y, Reg3y/Reg3x 

   Reg4x, Reg4y= np.sum(2*sobelx[0:13, 13:25]*sobely[0:13,13:25])+sys.float_info.epsilon, \
                 np.sum(sobelx[0:13,13:25]**2-sobely[0:13,13:25]**2)
   X_pix_data[k,12], X_pix_data[k,13], X_pix_data[k,14] =  Reg4x, Reg4y, Reg4y/Reg4x 
   Reg5x, Reg5y= np.sum(2*sobelx[13:25, 0:13]*sobely[13:25,0:13])+sys.float_info.epsilon, \
                 np.sum(sobelx[13:25,0:13]**2-sobely[13:25,0:13]**2)
   X_pix_data[k,15], X_pix_data[k,16], X_pix_data[k,17] =  Reg5x, Reg5y, Reg5y/Reg5x     
   Reg6x, Reg6y= np.sum(2*sobelx[13:25,13:25]*sobely[13:25,13:25])+sys.float_info.epsilon, \
                 np.sum(sobelx[13:25,13:25]**2-sobely[13:25,13:25]**2)    
   X_pix_data[k,18], X_pix_data[k,19], X_pix_data[k,20] =  Reg6x, Reg6y, Reg6y/Reg6x 

  X_pix_data_normalize = preprocessing.normalize(X_pix_data)
  pix_prediction = loaded_model.predict_classes(X_pix_data_normalize)
  pix_predict_proba = pd.DataFrame.from_records(loaded_model.predict_proba(X_pix_data_normalize))
  pix_prob_arr  = pix_predict_proba.as_matrix()
  pix_prob_list = pix_predict_proba.max(axis=1)

  for m in range(NoWidth):
   pix_dir[i][m] = pix_prediction[m]
   pix_dir_prob[i][m]  = pix_prob_list[m]
   pix_consensus[i][m]  = sum(weightTBL[pix_prediction[m]]*pix_prob_arr[m])
 print()
 sys.stdout.flush()
 pixtri=np.full(fm.shape, 192, dtype=int)
 #pixSMT=np.full(fm.shape, 192, dtype=int)

 pixSMT = MyL.UT_PixTri(plt, fpfg, pix_dir, pixtri, SMLoop=3)
 plt.title(sys.argv[2][-11:])
 plt.imshow(pixtri, cmap=plt.cm.gray)
 ImgPath='D:\\Fingerprint\\paper8_NN\\P8NN_Images\\'+sys.argv[1]+'_'+sys.argv[2][-11:-4]+"_7.png"
 plt.savefig(ImgPath, dpi=600, bbox_inches='tight')
 #plt.show()
 plt.title(sys.argv[2][-11:])
 plt.imshow(pixSMT, cmap=plt.cm.gray)
 ImgPath='D:\\Fingerprint\\paper8_NN\\P8NN_Images\\'+sys.argv[1]+'_'+sys.argv[2][-11:-4]+"_8.png"
 plt.savefig(ImgPath, dpi=600, bbox_inches='tight')
 #plt.show()







