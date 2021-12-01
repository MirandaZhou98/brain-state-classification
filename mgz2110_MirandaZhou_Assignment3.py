
import numpy as np
import nibabel as nib
from nilearn import image

import sklearn
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score

import scipy.io
import cv2


# Preprocessing

filename = './data/sub-01/ses-test/func/sub-01_ses-test_task-fingerfootlips_bold.nii'
re_filename = './data/sub-01/ses-retest/func/sub-01_ses-retest_task-fingerfootlips_bold.nii'

maskname = './data/sub-01/ses-test/func/c6sub-01_ses-test_task-fingerfootlips_bold.nii'
re_maskname = './data/sub-01/ses-retest/func/c6sub-01_ses-retest_task-fingerfootlips_bold.nii'

mat = scipy.io.loadmat('./data/label.mat')

label = mat['label']
labels = label.flatten()
# label 1 means Rest (fixation) which has 94 volumes
# label 2 means Finger movement which has 30 volumes
# label 3 means Lips movement which has 30 volumes
# label 4 means Foot movement which has 30 volumes


def preprocessing(filename, maskname):
    # return processed images with mask
    
    # loading image
    img_mask = nib.load(maskname)
    
    # open up file components
    test_img = []
    for volume in image.iter_img(filename):
        test_img.append(volume.dataobj)
        
    test_img = np.array(test_img)
    
    # binary mask with threshold
    th, dst = cv2.threshold(img_mask.get_fdata(),0.8,1, cv2.THRESH_BINARY_INV)
    
    # for each image in 184, match with corresponding mask. since mask is 0 and 1, just need to multiple all of them
    clean_test_img = test_img.copy()
    for i in range(len(test_img)):
        for j in range(dst.shape[2]):
            t_img = test_img[i,:,:,j]
            m_img = dst[:,:,j]
            new = t_img * m_img
            clean_test_img[i,:,:,j] = new
    clean_test_img = np.array(clean_test_img)
    return clean_test_img


def feature_selection(cleaned_images, n):
    # apply PCA to reshaped images
    
    # first decompose to matrix of 184 and combination of all other levels
    size, s1, s2, s3 = cleaned_images.shape
    reshaped_images = cleaned_images.reshape(size, s1*s2*s3)
    
    pca = PCA(n_components=n)
    pca_images = pca.fit_transform(reshaped_images)
    
    return reshaped_images, pca_images


def brain_state_classification(fname, mname):
    # main function to run everything

    cleaned_imgs = preprocessing(filename, maskname)
    reshaped_imgs, pca_imgs = feature_selection(cleaned_imgs, 100)

    # hyperparameter tuning
    param_grid = {'C': [0.0001, 0.001, 0.01, 0.1,1,10],
              'gamma': [0.00001, 0.0001,0.001,0.1,1],
              'kernel': ['linear','rbf','poly']}

    cross_val = StratifiedKFold(n_splits=9)

    svc = svm.SVC()
    clf = GridSearchCV(svc, param_grid, cv=cross_val)
    clf.fit(pca_imgs, labels)

    print('tuned hyperparameters for PCA')
    print('Best parameters: {}'.format(clf.best_params_))
    
    # test images
    svc_test = svm.SVC(kernel=clf.best_params_['kernel'], C=clf.best_params_['C'], gamma=clf.best_params_['gamma'])
    # PCA
    pca_mean = np.mean(cross_val_score(svc_test,pca_imgs,labels,cv=35))
    print("PCA mean:", pca_mean)
    # no PCA
    no_pca_mean = np.mean(cross_val_score(svc_test,reshaped_imgs,labels,cv=35))
    print("no PCA mean:", no_pca_mean)


print("on test images")
brain_state_classification(filename, maskname)
print("on re-test images")
brain_state_classification(re_filename, re_maskname)



