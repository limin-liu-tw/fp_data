import numpy as np

_DeBug_ = False
Smth  = 2
BGloop= 10
PixNo = 25
PixNo_1_2 = 12
DirNo = 15
DirAgl= [i*(180//DirNo) for i in range(DirNo)]
RowNo = lambda row: row//PixNo
ColNo = lambda col: col//PixNo

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

def UT_SetGray(plt, fpfg, fm, fpdir_prob, fpdir_porg):
   print('=Setting Prob. Image=')
   height, width = fm.shape[0], fm.shape[1] 
   NoHeight, NoWidth = RowNo(height), ColNo(width)
   fpdirpn = np.full((NoHeight,NoWidth), -1, dtype=float)
   cmax, cmin= 0, 255
   for i in range(fpfg.shape[0]):
     for j in range(fpfg.shape[1]): 
        a,b,c,d=i*PixNo,(i+1)*PixNo,j*PixNo,(j+1)*PixNo
        #print('setting', i,j, '=',a,b,c,d,':', fpfg[i][j], end=' ')
        clr=int(fpdir_prob[i][j]*255) if ( fpfg[i][j] ) else 172
        fm[a:b,c:d]=clr
        #print('Prob.',i,j,'....', fpdir_porg[i][j], fpdir_prob[i][j])
        if ( fpfg[i][j] ): 
           fpdirpn[i][j] = int(fpdir_prob[i][j]*255)
           print('{:.3f} {:3d} {:.3f}'.format(fpdir_porg[i][j]*100, clr, fpdir_prob[i][j]))
           if ( fpdirpn[i][j] > cmax ): cmax = fpdirpn[i][j]
           if ( fpdirpn[i][j] < cmin ): cmin = fpdirpn[i][j]
#   plt.imshow(fm, cmap=plt.cm.gray) 
#   plt.show()
#   cnor=158
#   if cmin<cnor: cmin=cnor
#   cdelta=cmax-cmin
#   print('cmax, cmin', cmax, cmin, cdelta)
#   for i in range(fpfg.shape[0]):
#     for j in range(fpfg.shape[1]): 
#       a,b,c,d=i*PixNo,(i+1)*PixNo,j*PixNo,(j+1)*PixNo
#       if ( fpfg[i][j] ): 
#         print('setting', i, j, 'old', fpdirpn[i][j] ,end=' new ')
#         if (fpdirpn[i][j]<cnor): 
#           fpdirpn[i][j]=0
#         else:
#           fpdirpn[i][j]=int((fpdirpn[i][j]-cmin)*255/cdelta)
#         print(fpdirpn[i][j])
#         fm[a:b,c:d]=int(fpdirpn[i][j])
#   plt.imshow(fm, cmap=plt.cm.gray) 
#   plt.show()          
   print('=Setting rob. Image....Done!')

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
         fm[a:b,c:d]=172
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




