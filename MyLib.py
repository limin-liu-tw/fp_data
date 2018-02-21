import numpy as np
import operator

_DeBug_ = False
Smth  = 2
BGloop= 10
PixNo = 25
PixNo_1_2 = 12
DirNo = 15
DirAgl= [i*(180//DirNo) for i in range(DirNo)]
RowNo = lambda row: row//PixNo
ColNo = lambda col: col//PixNo

def checkTri(angle, fac=0):
  if (fac==0):  # 0 +/- 30
   if ( angle >=30 and angle < 90):  #(60)
     return 0
   elif (angle >=90 and angle <150): #(120)
     return 128
   else:
     return 255 #(0)
  elif (fac==1):  # 30 +/- 30
   if ( angle >=0 and angle < 60):   #(30)
     return 0
   elif (angle >=60 and angle <120): #(90)
     return 128
   else:
     return 255  #(120)
  elif (fac==2):  # 10 +/- 30
   if ( angle >=40 and angle < 100): #(70)
     return 0 
   elif (angle >=100 and angle <160):#(130)
     return 128
   else:
     return 255  #(10)
  elif (fac==3):  # 20 +/- 30
   if ( angle >= 50 and angle < 110): #(80)
     return 0
   elif (angle >=110 and angle <170): #(140)
     return 128
   else: 
     return 255  #(20)
  elif (fac==4):  # 0 +/- 10 (60)
   if ( angle >=50 and angle < 70):  #(60)
     return 0
   elif (angle >=110 and angle <130): #(120)
     return 128
   elif ( angle <10 or angle > 170): #(0)
     return 255
   else:
     return 192  
  elif (fac==5):  # 40 +/- 20 (40)
   if ( angle >=20 and angle < 60): #(40)
     return 0
   elif (angle >=80 and angle <120): #(100)
     return 128
   elif ( angle >=140 and angle < 180): #(160)
     return 255
   else:
     return 192 
  elif (fac==6):  #70 +/- 20 (70)
   if ( angle >=50 and angle < 90):  #(70)
     return 0
   elif (angle >=110 and angle <150): #(130)
     return 128
   elif ( angle < 20 ): #(10)
     return 255
   else:
     return 192   

def prepWeight():
    wgt = np.empty(91)  # delta is b/w 0, 90
    for i in range(91): wgt[i] = (0.5+(np.cos(np.pi*((i/90)))/2))**6
    y_TrainW = np.zeros((180,180))
    for rw in range(180):
      y_TrainW[rw, rw]=1.0
      for i in range(1, 91):
        y_TrainW[rw,rw-i] = y_TrainW[rw,(rw+i)%180] = wgt[i]

    return y_TrainW 

def UT_Sobel(fgm, showData=False):
  Gx=fgm[0,2]*255.+2*255.*255.*fgm[1,2]+fgm[2,2]*255.-fgm[0,0]*255.-2*255.*255.*fgm[1,0]-fgm[2,0]*255.
  Gy=fgm[2,0]*255.+2*255.*255.*fgm[2,1]+fgm[2,2]*255.-fgm[0,0]*255.-2*255.*255.*fgm[0,1]-fgm[0,2]*255.
  if showData:
    print(fgm[0,:])
    print(fgm[1,:])
    print(fgm[2,:])
    print(Gx, Gy)
  return Gx, Gy

def FP_DirSobel(plt, fm, fbin, fpfg, showImg=False, dirNo = DirNo, pixNo=PixNo):
  print("Calculating direction Sobel ({0}), Window {1}x{1} Shape:{2}".format(dirNo, pixNo, fm.shape))
  height, width = fm.shape[0], fm.shape[1] 
  NoHeight, NoWidth = RowNo(height), ColNo(width)

  if showImg: axs=[[None for _ in range(NoWidth)]]*NoWidth
  fpdirSbl=np.full((NoHeight,NoWidth), -1, dtype=float)
  fpdirVGx=np.full((NoHeight,NoWidth), -1, dtype=float)
  for i in range(NoHeight):
    for j in range(NoWidth):
      #print('i, j:', i, j, 'shape:', fpdir.shape)
      a,b,c,d=i*PixNo,(i+1)*PixNo,j*PixNo,(j+1)*PixNo
      Cx, Cy, tmpagl, newagl, ctr = (a+b)//2, (c+d)//2, [[0]]*PixNo, [[0]]*DirNo, 0
      Gx, Gy, VGx, VGy = 0, 0, 0., 0.
      if fpfg[i][j]:  # we only process froeground
        #if i==3 and j==9: print(a,b,c,d)
        for gx in range(a+1, b-1):
          #print(gx, ':',end='')
          for gy in range(c+1, d-1):
            #print(gy, ' ')
            Gx, Gy=UT_Sobel(fm[gx-1:gx+2,gy-1:gy+2,0], False)
            VGx += 2*Gx*Gy
            VGy += (Gx**2-Gy**2)
        theta=(np.arctan(VGx/VGy))/2
        #if (theta < 0 and VGx>0): theta+=np.pi
        #if (theta < 0 and VGx<=0) or (theta>=0 and VGx>0):  theta+=np.pi/2
        #if i==3 and j==9: print('VGy/VGx=',VGy/VGx, 'arctan=', np.arctan2(VGy, VGx), 'theta=', theta)
        fpdirSbl[i][j]=theta
        fpdirVGx[i][j]=VGx
      else:
        fpdirSbl[i][j]=-1
      if i==3 and j==9: print(' {:.2f}'.format(fpdirSbl[i][j]))

  for i in range(NoHeight):
    print(i,':')
    for j in range(NoWidth):
      print(' {:.1f}'.format(fpdirSbl[i][j]), end='')
    print()

  for i in range(NoHeight):
    print(i,':')
    for j in range(NoWidth):
      if (fpdirSbl[i][j] < 0 and fpdirVGx[i][j]>0): fpdirSbl[i][j]+=np.pi
      if (fpdirSbl[i][j] < 0 and fpdirVGx[i][j]<=0) or (fpdirSbl[i][j]>=0 and fpdirVGx[i][j]>0):  fpdirSbl[i][j]+=np.pi/2
      print(' {:.1f}'.format(fpdirSbl[i][j]), end='')
    print()       
        
  UT_SetLine(plt, fpdirSbl, fm, axs, showImg, 'Red',3)
  if showImg: plt.imshow(fm, cmap=plt.cm.gray)
  if showImg: plt.show()
  print('==================')
  return fpdirSbl

def UT_SetTri(plt, fpfg, fpdir, fmdir, noColor=3):
   # fpfg: foreground(true)
   # fpdir: block direction
   
   print('=Setting Tri-color Image=')
   fmtri=fmdir.copy()
   print(fpfg.shape, fpdir.shape, fmtri.shape)
   height, width = fmtri.shape[0], fmtri.shape[1] 
   NoHeight, NoWidth = RowNo(height), ColNo(width)
  
   agl_lst = [(x*180)//noColor for x in range(noColor)]
   agl_d = 90//noColor
   clr_lst = [(x*256)//(noColor-1) for x in range(noColor)]
   clr_lst[-1]=clr_lst[-1]-1
   clr_delta=256/(noColor-1)

   for i in range(fpfg.shape[0]):
     for j in range(fpfg.shape[1]): 
        a,b,c,d=i*PixNo,(i+1)*PixNo,j*PixNo,(j+1)*PixNo
        if ( fpfg[i][j] ) :
            #print(i, j, fm[a:b,c:d] )
            for k in range(a,b):
             for l in range(c,d):
              fmtri[k,l]=192
            #fmtri[a:b,c:d].fill(192)#=192
           # print(fm[a:b,c:d])
        #else :
        #    fm[a:b,c:d]=255

        #fm[a:b,c:d]=clr
    
   print('=Setting Tri-color Image....Done!')
   return fmtri

def UT_PixTri(plt, fpfg, pixdir, pixtri, noColor=3, SWD=2, SMLoop=4):
   print('=Setting Pix-Tri. Image=')
   height, width = pixtri.shape[0], pixtri.shape[1] 
   #NoHeight, NoWidth = RowNo(height), ColNo(width)
   #fpdirpn = np.full((NoHeight,NoWidth), -1, dtype=float)
   print('pixdir.shape: ', pixdir.shape)
   print('pixtri.shape: ', pixtri.shape) 

   pixsmt=pixtri.copy()  # smooth base array
   W12 = 12
   for i in range(height):
     for j in range(width): 
       if i < W12 or i>= height - W12 or j < W12 or j>= width-12:
         pixtri[i][j], pixsmt[i][j]=192, -1
       else:
         if ( fpfg[i//PixNo][j//PixNo] ):
          pd=pixdir[i-W12][j-W12]
          pixtri[i-W12][j-W12] = pixsmt[i-W12][j-W12]=checkTri(pd,0)
          #if ( pd >= 30 and pd < 90 ):
          #  pixtri[i-W12][j-W12], pixsmt[i-W12][j-W12]=0, 0
          #elif ( pd >= 90 and pd < 150 ):
          #  pixtri[i-W12][j-W12], pixsmt[i-W12][j-W12]=128, 128
          #else:
          #  pixtri[i-W12][j-W12], pixsmt[i-W12][j-W12]=255, 255
         else:
            pixtri[i-W12][j-W12], pixsmt[i-W12][j-W12]=192, 192

   #SWD=2  # for 3x3 ; 5x5 needs 2
   th_no = ((2*SWD+1)**2)//2 + 1
   print('==== th_no =====' , th_no)
   for sm_no in range(SMLoop):
    pixbuf=pixsmt.copy()
    for i in range(height):
     for j in range(width): 
       if i < W12 or i>= height - W12 or j < W12 or j>= width-12: continue  
       zary = pixbuf[i-SWD,j-SWD:j+SWD+1].flatten()
       for m in range(2*SWD+1):
         if m == 0: continue
         mm = m - SWD
         for n in range(2*SWD+1):
           nn = n - SWD
           if mm == 0 and nn == 0: continue
           zary = np.append(zary, [pixbuf[i+mm, j+nn]])
       un, uc = np.unique(zary, return_counts=True)
       undict = dict(zip(un,uc))
       if undict[max(undict, key=undict.get)] >= th_no:
          pixsmt[i,j]=max(undict, key=undict.get)

   #plt.imshow(pixsmt, cmap=plt.cm.gray)
   #plt.show()
   print('=Setting Pix-Tri. Image....Done!')
   return pixsmt

def UT_SetTri(plt, fpfg, fpdir, fm, r_fac=0, noColor=3):
   print('=Setting Tri. Image=')
   height, width = fm.shape[0], fm.shape[1] 
   NoHeight, NoWidth = RowNo(height), ColNo(width)
   fpdirpn = np.full((NoHeight,NoWidth), -1, dtype=float)

   for i in range(fpfg.shape[0]):
     for j in range(fpfg.shape[1]): 
        a,b,c,d=i*PixNo,(i+1)*PixNo,j*PixNo,(j+1)*PixNo
        
        if ( fpfg[i][j] ):
          clr=checkTri(fpdir[i][j],0) 
          if ( clr == 0 ):
            fpdirpn[i][j] = 0
          elif (clr==128):
            fpdirpn[i][j] = 1
          elif (clr==255):
            fpdirpn[i][j] = 2 
          else:
            pass
        #if ( fpfg[i][j] ):
        #  if ( fpdir[i][j] >= 30 and fpdir[i][j] < 90 ):
        #     clr, fpdirpn[i][j] = 0, 0
        #  elif ( fpdir[i][j] >= 90 and fpdir[i][j] < 150 ):
        #     clr, fpdirpn[i][j] = 128, 1
        #  else:
        #     clr, fpdirpn[i][j] = 255, 2
        else:
          clr, fpdirpn[i][j] =192, -1
        #print('fm:', a, b, c, d)
        fm[a:b,c:d]=clr

   if r_fac != 0:   
    fm_fac = fm.copy()
    for i in range(0,fpfg.shape[0],r_fac):
     for j in range(0,fpfg.shape[1],r_fac): 
      a,b,c,d=i*PixNo,(i+r_fac)*PixNo,j*PixNo,(j+r_fac)*PixNo
      zarry = np.zeros((noColor+1,), dtype=np.int)
      for m in range(r_fac):
       if i+m >= fpfg.shape[0]: continue
       for n in range(r_fac):
        if j+n >= fpfg.shape[1]: continue
        clridx = int(fpdirpn[i+m][j+n])
        if clridx>=0:
          zarry[clridx]+=1
        else:
          zarry[noColor]+=1        
      #print (i, j, fpdirpn[i][j], fpdirpn[i][j+1],fpdirpn[i+1][j], fpdirpn[i+1][j+1], zarry, np.argmax(zarry) )
      #print (i, j, zarry, zarry[-1],np.argmax(zarry) )
      #if ( fpfg[i][j] ):
      if ( zarry[-1]==zarry[np.argmax(zarry)] ):
          clr=192
      elif ( np.argmax(zarry)==0 ):
          clr = 0
      elif ( np.argmax(zarry)==1 ):
          clr = 128
      elif ( np.argmax(zarry)==2 ):
          clr = 255
      else:
          clr =192
      #print('fm_fac:', a, b, c, d)
      fm_fac[a:b,c:d]=clr
     
   print('=Setting Tri. Image....Done!')
   if r_fac!=0:
     return fm_fac

def UT_SetGray(plt, fpfg, fm, fpdir_prob):
   print('=Setting Prob. Image=')
   height, width = fm.shape[0], fm.shape[1] 
   NoHeight, NoWidth = RowNo(height), ColNo(width)
   fpdirpn = np.full((NoHeight,NoWidth), -1, dtype=float)
   cmax, cmin= 0, 255
   for i in range(fpfg.shape[0]):
     for j in range(fpfg.shape[1]): 
        a,b,c,d=i*PixNo,(i+1)*PixNo,j*PixNo,(j+1)*PixNo
        #print('setting', i,j, '=',a,b,c,d,':', fpfg[i][j], end=' ')
        clr=int(fpdir_prob[i][j]*255) if ( fpfg[i][j] ) else 192
        fm[a:b,c:d]=clr
        #print('Prob.',i,j,'....', fpdir_porg[i][j], fpdir_prob[i][j])
        if ( fpfg[i][j] ): 
           fpdirpn[i][j] = int(fpdir_prob[i][j]*255)
           print('{} {} {:3d} {:.3f}'.format(i, j, int(fpdir_prob[i][j]*255), fpdir_prob[i][j]))
         
   print('=Setting rob. Image....Done!')

def UT_SetLine2(plt, fpfg, fpdir, fmS, showImg=False, Clr='white', Width=3):
   print('=Setting Line Segment=', Clr, Clr=='Red')
   print(fmS)
   fm=np.full(fmS, 255, dtype=int)
   CC = 255 if Clr == 'white' else 0
   CW = 0
   CB = int(128)
   for i in range(fpfg.shape[0]):
     for j in range(fpfg.shape[1]): 
        a,b,c,d=i*PixNo,(i+1)*PixNo,j*PixNo,(j+1)*PixNo
        Cx, Cy, tmpagl, newagl, ctr = (a+b)//2, (c+d)//2, [[0]]*PixNo, [[0]]*DirNo, 0
        CxX, CyY=0,0
        if ( fpfg[i][j]):

         if (fpdir[i][j]<=45 or fpdir[i][j] >=135):
          CxOff = PixNo_1_2 * np.tan(fpdir[i][j]*np.pi/180)
          # to point
          CxX, CyY=int(Cx-CxOff), int(Cy+PixNo_1_2)          
          # from point
          _CxX, _CyY = Cx*2-CxX,Cy*2-CyY
          RGB = 0 if Clr=='Red' else 2 if Clr=='Blue' else 1
          for _r in range(PixNo):
             _x_, _y_ = int(_CxX-(_r * np.tan(fpdir[i][j]*np.pi/180))), _CyY+_r
             if Width==5:
               fm[_x_,_y_]=fm[_x_-1,_y_]=fm[_x_-2,_y_]=fm[_x_+1,_y_]=fm[_x_+2,_y_]=CC
             else:
               fm[_x_,_y_]=fm[_x_-1,_y_]=fm[_x_+1,_y_]=CC

         else:
          CyOff = PixNo_1_2 / np.tan(fpdir[i][j]*np.pi/180)
          CxX, CyY=int(Cx-PixNo_1_2), int(Cy+CyOff)
          _CxX, _CyY = Cx*2-CxX, Cy*2-CyY
          RGB = 0 if Clr=='Red' else 2 if Clr=='Blue' else 1
          for _r in range(PixNo):
             _x_, _y_ = _CxX-_r, int(_CyY+ (_r/np.tan(fpdir[i][j]*np.pi/180)))
             if Width==5:
               fm[_x_,_y_]=fm[_x_,_y_-1]=fm[_x_,_y_-2]=fm[_x_,_y_+1]=fm[_x_,_y_+2]=CC
             else:
               fm[_x_,_y_]=fm[_x_,_y_-1]=fm[_x_,_y_+1]=CC

        else:
         fm[a:b,c:d]=int(192)

   print('=Setting Line Segment....Done!')
   return fm

def UT_SetLine(plt, fpfg, fpdir, fm, axs, showImg=False, Clr='white', Width=3):
   print('=Setting Line Segment=', Clr, Clr=='Red')
   CC = 255 if Clr == 'white' else 0
   for i in range(fpdir.shape[0]):
     for j in range(fpdir.shape[1]): 
        a,b,c,d=i*PixNo,(i+1)*PixNo,j*PixNo,(j+1)*PixNo
        Cx, Cy, tmpagl, newagl, ctr = (a+b)//2, (c+d)//2, [[0]]*PixNo, [[0]]*DirNo, 0
        CxX, CyY=0,0
        if ( fpfg[i][j]):
         if (fpdir[i][j]<=45 or fpdir[i][j] >=135):
          CxOff = PixNo_1_2 * np.tan(fpdir[i][j]*np.pi/180)
          # to point
          CxX, CyY=int(Cx-CxOff), int(Cy+PixNo_1_2)          
          # from point
          _CxX, _CyY = Cx*2-CxX,Cy*2-CyY
          RGB = 0 if Clr=='Red' else 2 if Clr=='Blue' else 1
          #print('i,j ({},{}), a,b ({},{}), from ({},{}) to ({},{}), dir({})'.format(i,j,a,b,_CxX, _CyY,CxX, CyY,fpdir[i][j]))
          for _r in range(PixNo):
             _x_, _y_ = int(_CxX-(_r * np.tan(fpdir[i][j]*np.pi/180))), _CyY+_r
             #_x_, _y_ = int(a-(_r * np.tan(fpdir[i][j]*np.pi/180))), _CyY+_r
             if Width==5:
               fm[_x_,_y_]=fm[_x_-1,_y_]=fm[_x_-2,_y_]=fm[_x_+1,_y_]=fm[_x_+2,_y_]=CC
             else:
               fm[_x_,_y_]=fm[_x_-1,_y_]=fm[_x_+1,_y_]=CC

         else:
          CyOff = PixNo_1_2 / np.tan(fpdir[i][j]*np.pi/180)
          CxX, CyY=int(Cx-PixNo_1_2), int(Cy+CyOff)
          _CxX, _CyY = Cx*2-CxX, Cy*2-CyY
          #from (CxX,CyY) -> (Cx*2-CxX,Cy*2-CyY), plot one red pixles
          RGB = 0 if Clr=='Red' else 2 if Clr=='Blue' else 1
          #print('i,j ({},{}), a,b ({},{}), from ({},{}) to ({},{}), dir({})'.format(i,j,a,b,_CxX, _CyY,CxX, CyY,fpdir[i][j]))
          for _r in range(PixNo):
             _x_, _y_ = _CxX-_r, int(_CyY+ (_r/np.tan(fpdir[i][j]*np.pi/180)))
             if Width==5:
               fm[_x_,_y_]=fm[_x_,_y_-1]=fm[_x_,_y_-2]=fm[_x_,_y_+1]=fm[_x_,_y_+2]=CC
             else:
               fm[_x_,_y_]=fm[_x_,_y_-1]=fm[_x_,_y_+1]=CC
        else:
         fm[a:b,c:d]=192
        if showImg: axs[i][j]=plt.subplot2grid((fpdir.shape[0],fpdir.shape[1]),(i,j))

        if showImg: axs[i][j].imshow(fm[a:b,c:d], cmap=plt.cm.gray)
        #if showImg and fpdir[i][j]!=-1: axs[i][j].plot((Cx,Cy),(Cx*2-CxX,Cy*2-CyY), color="red", linewidth=2.0, linestyle="-" )
        if showImg: axs[i][j].set_axis_off()
    #print('i, j:', i, j, 'Dir:', fpdir[i])
    #fpdir[i][j]=0
   if showImg: plt.axis('off')
   if showImg: plt.show()
   print('=Setting Line Segment....Done!')

UT_WaveFrq_buffer=np.full((DirNo,3), 0, dtype=float)
def UT_WaveFrq(fYs, showData=False):
  res=0

  for i in range(fYs.shape[0]):
     UT_WaveFrq_buffer[i][0], UT_WaveFrq_buffer[i][1]=np.average(fYs[i]), np.std(fYs[i])

  UT_WaveFrq_buffer[0,2]=(UT_WaveFrq_buffer[DirNo-1,1]+UT_WaveFrq_buffer[0,1]+UT_WaveFrq_buffer[1,1])/3
  for i in range(1, fYs.shape[0]-1):
     UT_WaveFrq_buffer[i,2]=np.average(UT_WaveFrq_buffer[i-1:i+2,1])
  UT_WaveFrq_buffer[DirNo-1,2]=(UT_WaveFrq_buffer[DirNo-1,1]+UT_WaveFrq_buffer[0,1]+UT_WaveFrq_buffer[DirNo-2,1])/3
  #a = np.array(UT_WaveFrq_buffer[:,2])  # smooth std does not help
  a = np.array(UT_WaveFrq_buffer[:,1]) 
  if ( showData ) : print('===>', a, '\n', UT_WaveFrq_buffer)
  return np.where(a == a.min())[0][0]

def FP_FindDir(plt, fm, fbin, fpfg, showImg=False, dirNo = DirNo, pixNo=PixNo):
  print("Calculating direction Frequency ({0}), Window {1}x{1} Shape:{2}".format(dirNo, pixNo, fm.shape))
  height, width = fm.shape[0], fm.shape[1] 
  NoHeight, NoWidth = RowNo(height), ColNo(width)

  if showImg: axs=[[None for _ in range(NoWidth)]]*NoWidth
  fpdir=np.full((NoHeight,NoWidth), -1, dtype=float)
  fYs  =np.full((DirNo,PixNo), 0, dtype=float)
  #print('anx:', axs, '\n', DirNo, ':', DirAgl, '\n', fpdir, 'H/W', NoHeight, NoWidth)
  for i in range(NoHeight):
    for j in range(NoWidth):
      #print('i, j:', i, j, 'shape:', fpdir.shape)
      a,b,c,d=i*PixNo,(i+1)*PixNo,j*PixNo,(j+1)*PixNo
      Cx, Cy, tmpagl, newagl, ctr = (a+b)//2, (c+d)//2, [[0]]*PixNo, [[0]]*DirNo, 0
      
      if fpfg[i][j]:  # we only process froeground
         for agl in DirAgl:
           newagl[ctr] = [[0]]*PixNo
           if ( agl<=45 or agl >=135):
             ta = np.tan(agl*np.pi/180)
             tmpagl=[(x, int(x*ta)) for x in range(PixNo)]
             CxOff, CyOff= -tmpagl[PixNo_1_2][0], -tmpagl[PixNo_1_2][1]
             for x in range(PixNo):
               newagl[ctr][x]=[Cx-(tmpagl[x][1]+CyOff), Cy+(tmpagl[x][0]+CxOff)]
           else:
             ta = np.tan(agl*np.pi/180)
             tmpagl=[(int(y/ta),y) for y in range(PixNo)]
             CxOff, CyOff= -tmpagl[PixNo_1_2][0], -tmpagl[PixNo_1_2][1]
             for y in range(PixNo):
               newagl[ctr][y]=[Cx-(tmpagl[y][1]+CyOff), Cy+(tmpagl[y][0]+CxOff)]
 
           # for every angle place it's fY[i]
           for idx in range(PixNo):
               ia, ib = newagl[ctr][idx][0], newagl[ctr][idx][1]
               fYs[ctr][idx] = fm [ia] [ib] [0]  
           ctr+=1                     
         fpdir[i][j] = DirAgl[UT_WaveFrq(fYs)]

  UT_SetLine(plt, fpdir, fm, axs, showImg, 'Red',5)
  return fpdir         
  
def FP_FindBG(plt, fm, showImg=False, pixNo=PixNo):
  #fdir=np.copy(fm)
  print("Finding foreground image, Window {0}x{0} Shape:{1}".format(pixNo, fm.shape))
  height, width = fm.shape[0], fm.shape[1] 
  NoHeight, NoWidth = RowNo(height), ColNo(width)
  if showImg: axs=[[None for _ in range(NoWidth)]]*NoHeight
  aavg, astd =np.average(fm), np.std(fm) 
  if _DeBug_: print("...average{} std{}".format(aavg,astd))
  # foreground image
  fpfg=np.ndarray((NoHeight,NoWidth),  dtype=bool)
  for i in range(NoHeight):
    if _DeBug_: print("i:({:2d})".format(i),end='')
    for j in range(NoWidth):
        a,b,c,d=i*PixNo,(i+1)*PixNo,j*PixNo,(j+1)*PixNo
        ravg=np.average(fm[a:b,c:d])
        rstd=np.std(fm[a:b,c:d])
        if _DeBug_: print(",{:2d},{:2d}".format(int(ravg*100),int(rstd*100)),end='')
        fpfg[i][j] = False if ravg > aavg + astd/4 else True
        if not fpfg[i][j]:
         for m in range(a,b):
          for n in range(c,d):  
           fm[m][n]=0 
        if showImg: axs[i][j]=plt.subplot2grid((NoHeight,NoWidth),(i,j))
        if showImg: axs[i][j].imshow(fm[a:b,c:d], cmap=plt.cm.gray)
        if showImg: axs[i][j].set_axis_off()
    if _DeBug_: print('')
  # remove island
  for k in range(BGloop):
   for i in range(NoHeight):
    for j in range(NoWidth):
      # check if this blk is really a forground
      if fpfg[i][j]:
        count=0;
        for m in range(i-1,i+2):
         for n in range(j-1,j+2):
           if m==i and n==j: continue
           if m<0 or m==NoHeight:
             count-=1.1
             continue
           elif n<0 or n == NoWidth:
             count-=1.1
             continue
           else:
             count=count+1 if fpfg[m][n] else count-1
        if count<0:
          fpfg[i][j]=False
          a,b,c,d=i*PixNo,(i+1)*PixNo,j*PixNo,(j+1)*PixNo
          for m in range(a,b):
            for n in range(c,d):  
              fm[m][n]=1  
          
  if showImg: plt.axis('off')
  if showImg: plt.show()
  if showImg: plt.imshow(fm,cmap=plt.cm.gray)
  if showImg: plt.show()
  return fpfg

def FP_Binary(plt, fm, showImg=False):
  global fbin
  fbin=np.copy(fm)
  print("Binarizing gray scale image ", fm.shape, " window", PixNo)
  avg=np.average(fm[:,:])
  std=np.std(fm[:,:])
  height, width = fm.shape[0], fm.shape[1]
  NoHeight, NoWidth = RowNo(height), ColNo(width)

  if showImg: axs=[[None for _ in range(NoWidth)]]*NoHeight
  #plt.axis('off')
  th=np.empty([NoHeight,NoWidth], dtype = float)  
  for i in range(NoHeight):
    #print(i,'= ', end='')
    for j in range(NoWidth):
      #if fpfg[i][j]: 
        a,b,c,d=i*PixNo,(i+1)*PixNo,j*PixNo,(j+1)*PixNo
        ravg=np.average(fm[a:b,c:d])
        rstd=np.std(fm[a:b,c:d])
        w=0
        if ravg > avg and rstd < 0.1:
          b_avg,w = avg + std/3, 1
        elif ravg > avg and rstd >= 0.1:
          if ravg - avg < std/3:
            if rstd < 0.15:
               b_avg, w = avg + std/4, 2.1
            else:
               b_avg, w = ravg - std, 2.2 # + std/4 + 0.1, 2.2
          else:
            if ravg > 0.75:
               if ( rstd > 0.15 ) :
                  b_avg, w = avg - std/3 , 2.3
               else:
                  b_avg, w = avg + std/4 , 2.4
            else:
               b_avg, w = avg , 2.5
        elif ravg > (avg - std) :
          b_avg, w = (ravg + (avg - std/2) )/2, 3
        else:
          b_avg, w = (ravg + (avg - std) )/2, 4
        th[i][j]=b_avg

  for i in range(NoHeight):
    for j in range(NoWidth):
        a,b,c,d=i*PixNo,(i+1)*PixNo,j*PixNo,(j+1)*PixNo
        if showImg:axs[i][j]=plt.subplot2grid((NoHeight,NoWidth),(i,j))

        for m in range(a,b):
          for n in range(c,d):  
              fbin[m][n]=0 if fm[m][n] < th[i][j] else 1
        if showImg: axs[i][j].imshow(fbin[a:b,c:d], cmap=plt.cm.gray)
        if showImg: axs[i][j].set_axis_off()
  if showImg: plt.imshow(fbin)
  if showImg: plt.axis('off')
  if showImg: plt.show()
  return plt, fbin

def FP_Smooth(fdata):
  global fm
  fm=np.copy(fdata)
  print("Smoothing gray scale image ", fm.shape, " window", PixNo)
  height, width = fdata.shape[0], fdata.shape[1] 
  for r in range(Smth, height-Smth):
     for c in range(Smth, width-Smth):
        total= 0
        ## do smooth here
        for x in range(r-Smth, r+Smth+1):
            for y in range(c-Smth,c+Smth+1):
                total+=fdata[x][y];
        fm[r][c]=total/((Smth*2+1)**2)
  return fm






