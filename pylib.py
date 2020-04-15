#!/usr/bin/env python3
import numpy as np
import random
import math
from netCDF4 import Dataset
from scipy import signal
import imageio
from skimage import segmentation as seg
import matplotlib.pyplot as plt
from PIL import Image
from skimage import transform
from collections import Counter
from scipy import signal, ndimage
from skimage import morphology as mor
import time


#================================Save Data==================================
def Save_as_NC(path, data, tomo_or_seg):
    datatype=data.dtype
    ncw=Dataset(path, 'w', format='NETCDF4')
    layer=data.shape[0]
    row=data.shape[1]
    col=data.shape[2]
    x=ncw.createDimension('x', layer) #***************            
    y=ncw.createDimension('y', row)
    z=ncw.createDimension('z', col)
    Coorx=np.arange(0, layer, 1)
    Coory=np.arange(0, row, 1)
    Coorz=np.arange(0, col, 1)
    x_data=ncw.createVariable('x', 'i8', ('x',))
    y_data=ncw.createVariable('y', 'i8', ('y',))
    z_data=ncw.createVariable('z', 'i8', ('z',))
    x_data=Coorx
    y_data=Coory
    z_data=Coorz
    if tomo_or_seg==0:
        data1=ncw.createVariable('tomo', datatype, ('x', 'y', 'z')) # Note the format of the data, here is 'i4'
    else:
        data1=ncw.createVariable('segmented', datatype, ('x', 'y', 'z')) # Note the format of the data, here is 'i4'
    data1[:]=data
    ncw.close()

def Save_as_png(path, data, filename):
    dimension=data.shape
    minvalue=np.min(data)
    maxvalue=np.max(data)
    data=255*((data-minvalue)/(maxvalue-minvalue))
    data.astype('u1')
    if len(dimension)==2:
        if filename=='':
            imageio.imwrite(path+'/'+'new_image.png', data)
        else:
            imageio.imwrite(path+'/'+filename, data)
    if len(dimension)==3:
        z=dimension.shape[0]
        length=len(str(z))
        for i in rang(z):
            k=str(i)
            k1=k.zfill(length+1)
            imageio.imwrite(path+'/'+filename+'_'+k1+'.png', data[i, :, :])            
       
def Save_as_tiff(path, data, filename):
    #tiff only save uint8 format
    if data.dtype!='uint8':
        data.astype('f4')
        data=255*(data-np.min(data))/(np.max(data)-np.min(data))
        data=data.astype('u1') 
        print(data.dtype, np.max(data), np.min(data))       
    dimension=data.shape
    if len(dimension)==2:
        if filename=='':
            imageio.imwrite(path+'/'+'new_image.tiff', data)
        else:
            imageio.imwrite(path+'/'+filename+'.tiff', data)
    if len(dimension)==3:
        z=dimension[0]
        length=len(str(z))
        for i in range(z):
            k=str(i)
            k1=k.zfill(length+1)
            imageio.imwrite(path+'/'+filename+'_'+k1+'.tiff', data[i, :, :]) 

def Format_Transform_Raw_to_NC(path, dtype, tomo_or_seg, row, col, layer):
    data=np.fromfile(path, dtype=dtype)
    data=data.reshape(layer,row,col)
    filename=os.path.basename(path)
    filename=filename.split('.')[0]
    savepath=os.path.dirname(path)+'/'+filename+'.nc'
    Save_as_NC(savepath, data, tomo_or_seg)
 

if __name__=='__main__':

    nc=Dataset('/home/wangy0z/yzw/Data/Sandstone/Partition_Grains_test.nc', 'r')
    data=nc.variables['segmented']
    Save_as_tiff('/home/wangy0z/yzw/Data/Sandstone/Partition_grains_test_tiff', data, 'sythetic')





















