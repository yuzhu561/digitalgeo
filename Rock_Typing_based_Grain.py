#!/usr/bin/env python3
import numpy as np
import random
import math
from netCDF4 import Dataset
from scipy import signal, ndimage
import imageio
import os
import sys
from skimage import morphology as mor
from skimage.feature import peak_local_max
from skimage import filters
import matplotlib.pyplot as plt
from scipy import optimize
import scipy.spatial.distance as dis
from sklearn import mixture, svm
from scipy.spatial import KDTree
from collections import Counter

Current_dir=os.getcwd()
import pylib

eps = np.finfo(np.float).eps

def Grain_Partition_Erosion_Dilation(seg, target_phase, its_max, remove_small_object_or_not, save_path):
    img=np.zeros(seg.shape, dtype='u1')
    idx=np.flatnonzero(seg==target_phase)
    img.flat[idx]=1
    distance=ndimage.distance_transform_edt(img) 

    structure=np.ones((3,3,3), dtype='u1')
    #structure=mor.ball(1)
    img1=img.copy()
    img2=img.copy()
    label_1, num_1=ndimage.label(img1, structure)
    location_1=[]
    for i in range(num_1):
        idx_i=np.flatnonzero(label_1==i+1)
        location_1.append(idx_i)
    its=0
    while (its<its_max):
        img1=mor.binary_erosion(img1, structure)
        label, num=ndimage.label(img1, structure)

        for p in location_1:
            if sum(label.flat[p])==0:
                img1.flat[p]=1
        img1=img1.astype('u1')
        label_1, num_1=ndimage.label(img1, structure)

        location_1=[]
        for i in range(num_1):
            idx_i=np.flatnonzero(label_1==i+1)
            location_1.append(idx_i)
        its=its+1
   
    img_after_erosion=label_1>0
    img_after_erosion=img_after_erosion.astype('i8')
    mass_center=ndimage.measurements.center_of_mass(img_after_erosion, label_1, list(range(1, num_1+1)))
    mass_center=np.array(mass_center)
    mass_center=mass_center.astype('i8')
    print(len(mass_center)) 
    markers=np.zeros(distance.shape, dtype='i8')
    m=1
    for i in range(mass_center.shape[0]):
        markers[mass_center[i][0]][mass_center[i][1]][mass_center[i][2]]=m
        m=m+1 
            
    labels = mor.watershed(-distance, markers, connectivity=np.ones((3, 3, 3)), mask=img)
    print(len(np.unique(labels)))

    #===============Remove small objects=================
    if remove_small_object_or_not==1:
        markers1=markers.copy()
        markers1=np.array(markers1>0).astype('u1')
        uni=list(np.unique(labels))
        uni=uni[1:len(uni)]
        label_pus=len(uni)+2
        for i in uni:
            idx=np.flatnonzero(labels==i)
            if len(idx)<100:
                tt=np.zeros(markers1.shape, dtype='u1')
                tt.flat[idx]=1
                tt1=mor.binary_dilation(tt, structure)
                idx1=np.nonzero(np.logical_and(tt1==1, tt==0))
                idx1=np.ravel_multi_index(idx1, markers1.shape)
                neighbours=np.unique(labels.flat[idx1])
                if len(neighbours)==1:
                    labels.flat[idx]=neighbours[0]
                else:
                    labels.flat[idx]=neighbours[1]

    #===============Save the image============================
    if len(save_path)>0:
        pylib.Save_as_NC(save_path, labels, 1)
    
    '''
    fig, axes = plt.subplots(ncols=2, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(labels[50,:,:], cmap=plt.cm.nipy_spectral)
    ax[0].set_title('slice_20')
    ax[1].imshow(labels[:,100,:], cmap=plt.cm.nipy_spectral)
    ax[1].set_title('slice_30')
    plt.show()
    '''

    return labels

def Rock_typing_objects_based(img1, windowsize, num_type, training):
    #img1 is the image after grain partitioning with n grains and each grain has an unique label such as 100.
    #windowsize is an integer which defines the side length of the scanning window
    #num_type is a integer which defines how many rock types are expected to divided
    #training is a list with two elements where training[0] is a list contains the labels of the grains manually identified and training[1] contains its corresonding rock types

    grains=np.unique(img1) #record the elements of the input image
    grains=list(grains[1:len(grains)]) # remove the 0 phase
    
    Z=img1.shape[0]
    Y=img1.shape[1]
    X=img1.shape[2]

    structure=np.ones(np.array([3,3,3]), dtype='u1')

    hw=int(windowsize/2.0)
    boundary_extend=(-1)*(np.ones((Z+2*hw, Y+2*hw, X+2*hw), dtype='i4'))
    boundary_extend[hw:Z+hw, hw:Y+hw, hw:X+hw]=img1

    boundary_extend1=np.zeros((Z+2*hw, Y+2*hw, X+2*hw), dtype='i4')
    boundary_extend1[hw:Z+hw, hw:Y+hw, hw:X+hw]=img1
    
    img2=(-1)*(np.ones((Z+2*hw, Y+2*hw, X+2*hw), dtype='i1'))
    Bimg=np.zeros((Z+2*hw, Y+2*hw, X+2*hw), dtype='u4')
    Bimg[hw:Z+hw, hw:Y+hw, hw:X+hw]=img1
    mass_center=ndimage.measurements.center_of_mass(Bimg, boundary_extend1, grains)
    mass_center=np.array(mass_center)
    mass_center=mass_center.astype('i8')

    grain_vectors1=[]
    Training_features=[]
    Training_labels=[]
    its=0
    for i in grains:
        z=mass_center[its][0]
        y=mass_center[its][1]
        x=mass_center[its][2]

        idx=np.flatnonzero(boundary_extend==i)
        grain_size=len(idx) #calculate the size of the target grain (number of voxels)


        subset=boundary_extend[z-hw:z+hw+1, y-hw:y+hw+1, x-hw:x+hw+1]
        idx_within_boundary=np.flatnonzero(subset>=0)
        idx_within_boundary_0=np.flatnonzero(subset==0)
        phi=np.float(len(idx_within_boundary_0))/np.float(len(idx_within_boundary)) #calculate the local porosity

        neighbours=np.unique(subset)
        zero_location=np.where(neighbours==0)
        if len(zero_location[0])>0 and len(neighbours)>2:
            neighbours=neighbours[1:-1]
        neighbours=list(neighbours)
        neighbourhood=[]
        for j in neighbours:
            idx_1=np.flatnonzero(boundary_extend==j)
            current_object_size=len(idx_1)
            neighbourhood.append(current_object_size)
        mean_size=np.mean(neighbourhood)
        std_var_size=np.std(neighbourhood)


        #calculate the sphericity and the surface-volume-ratio of the grain (selected)===============
        trans=np.zeros(boundary_extend.shape, dtype='u1')
        trans.flat[idx]=1
        trans_dilation=mor.binary_dilation(trans, structure)
        idxtrans=np.nonzero(np.logical_and(trans_dilation==1, trans==0))
        idxtrans=np.ravel_multi_index(idxtrans, boundary_extend.shape)
        sphericity=(3.14159**(1/3.0))*((6*grain_size)**(2/3.0))/len(idxtrans) #calculate the sphericity of the target grain
        surface_volume_ratio=len(idxtrans)/np.float(grain_size) #calculate the surface_volume_ratio of th etarget grain
 
        grain_vectors1.append([grain_size, mean_size, std_var_size, sphericity, surface_volume_ratio])
        if i in training[0]:
            Training_features.append(grain_vectors1[its])
            location_label=np.where(np.array(training[0])==i)
            location_label=location_label[0][0]
            Training_labels.append(training[1][location_label])
        its=its+1

    #save the feature vetors==========================================================
    grain_features=np.zeros((len(grain_vectors1), 5), dtype='f8')
    its=0
    for i in grain_vectors1:
        grain_features[its, :]=np.array(i) 
        its=its+1

    training_data=np.zeros((len(Training_features), 6), dtype='f8')
    for i in range(len(Training_features)):
        training_data[i, 0:5]=np.array(Training_features[i]) 
        training_data[i][5]=Training_labels[i]
    np.savetxt('/home/features.txt', grain_features) #******************************************************    
    np.savetxt('/home/training_features.txt', training_data) #**********************************************

    #============Using SVM to classify the grains into different types================
    clf=svm.SVC(gamma='scale')
    clf.fit(Training_features,Training_labels)
    predicted_grain=clf.predict(grain_vectors1)

    for i in range(len(grains)):
        idx=np.flatnonzero(boundary_extend==grains[i])
        img2.flat[idx]=predicted_grain[i]

    img2=img2[hw:Z+hw, hw:Y+hw, hw:X+hw]

    pylib.Save_as_NC('/home/GMM_Modify_svm_3.nc', img2, 1) #*********************
    #===================identify the rock type by dilation=======================
    idx_0=np.flatnonzero(img2==-1)
    distance_matrix=np.zeros((num_type, len(idx_0)), dtype='i8')
    for i in range(num_type):
        current_type=np.zeros(img2.shape, dtype='i4')
        idex=np.flatnonzero(img2!=i)
        current_type.flat[idex]=1
        distance=ndimage.distance_transform_edt(current_type)
        distance_matrix[i, :]=distance.flat[idx_0]
    #print('distance_matrix is', distance_matrix[:, 10000:12000])
    location=np.argmin(distance_matrix, axis=0) 
    img2.flat[idx_0]=location

    pylib.Save_as_NC('/home/GMM_Modify_svm_3_fill.nc', img2, 1) #*****************

    #===========Remove the small isolated objects==============================
    for i in range(num_type):
        current_type=np.zeros(img2.shape, dtype='i4')
        idex=np.flatnonzero(img2==i)       
        current_type.flat[idex]=1
        labels, num=ndimage.label(current_type, structure)
        current_objects=list(np.unique(labels))
        current_objects=current_objects[1:len(current_objects)]
        for j in current_objects:
            idx_j=np.flatnonzero(labels==j)
            clusters=img1.flat[idx_j]
            clusters_ele=list(np.unique(clusters))
            clusters_ele=clusters_ele[1:len(clusters_ele)]
            if len(clusters_ele)<2:
                boundary_search=np.zeros(img2.shape, dtype='i4')
                boundary_search.flat[idx_j]=1
                boundary_search_dilation=mor.binary_dilation(boundary_search, structure)
                idxtrans=np.nonzero(np.logical_and(boundary_search_dilation==1, boundary_search==0))
                idxtrans=np.ravel_multi_index(idxtrans, img2.shape)
                boundary_ele=img2.flat[idxtrans] 
                ele_idx=Counter(boundary_ele).most_common(1)
                img2.flat[idx_j]=boundary_ele[ele_idx[0][0]]                              

    pylib.Save_as_NC('save_path', img2, 1) #**********************

    return img2

def modify(img2, minimum_volume):
    structure=np.ones(np.array([3,3,3]), dtype='u1')
    img2=img2+1
    plt.imshow(img2[100,:,:])
    plt.show()
    phases=list(np.unique(img2))
    print(phases)    
    phases=phases[1:len(phases)]
    print(phases)
    for i in phases:
        idx=np.flatnonzero(img2==i) 
        trans=np.zeros(img2.shape, dtype='u1')
        trans.flat[idx]=1
        labels, num=ndimage.label(trans, structure)
        print('number of objects is', num)
        if num>1:
            objects=list(np.unique(labels))
            objects=objects[1:len(objects)]
            print('objects are', objects)
            for j in objects:
                index_trans1=np.flatnonzero(labels==j)
                if len(index_trans1)<minimum_volume:
                    trans1=np.zeros(img2.shape, dtype='u1')
                    trans1.flat[index_trans1]=1
                    trans1_dilation=mor.binary_dilation(trans1, structure)
                    idxtrans=np.nonzero(np.logical_and(trans1_dilation==1, trans1==0))
                    idxtrans=np.ravel_multi_index(idxtrans, labels.shape)
                    current_boundary=np.unique(img2.flat[idxtrans])
                    print('current neighbours are', current_boundary)
                    zero_location=np.where(current_boundary==0)
                    if len(zero_location[0])>0:
                        current_boundary=current_boundary[1:len(current_boundary)]
                    if len(set(current_boundary))==1:
                        img2.flat[index_trans1]=list(set(current_boundary))[0]

    pylib.Save_as_NC('save_pathe/output.nc', img2, 1) #*********************************************

    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(img2[:,50,:], cmap=plt.cm.nipy_spectral)
    ax[0].set_title('slice_100')
    ax[1].imshow(img2[:,:,100], cmap=plt.cm.nipy_spectral)
    ax[1].set_title('slice_100')
    ax[2].imshow(img2[80,:,:], cmap=plt.cm.nipy_spectral)
    ax[2].set_title('slice_100')
    plt.show() 
    return img2   

if __name__ == "__main__":

    ncw = Dataset('original_image_path', 'r')
    
    x1=[685,808,963] #example of grain labels of rock type 1 (should be more)
    y1=[2012, 1374, 1970] #example grain labels of rock type 2 (should be more)
    
    x1=list(np.unique(np.array(x1)))
    x2=list(np.unique(np.array(x2)))
    y1=list(np.zeros(len(x1), dtype='f4'))
    y2=list(np.ones(len(x2), dtype='f4'))
    x=x1+x2
    y=y1+y2
    tarining_data=[x, y]

    data1=ncw.variables['segmented']
    data1=np.array(data1)
    data2=data1
    Rock_typing_objects_based(data2, 30, 2, tarining_data)  







