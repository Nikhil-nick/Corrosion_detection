# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:43:15 2020

@author: Pavan
"""

import SimpleITK as sitk
import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage.feature import greycomatrix, greycoprops
import pandas as pd

image=cv2.imread(r"C:\Users\Pavan\Desktop\Project\image_1.jpg")
#image = image.resize((100, 100))
#result=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grayscale = rgb2gray(image)
#g_e= greycomatrix(grayscale, [1],[0], 256, symmetric=False, normed=True)
grayscale = np.array(grayscale)
im = sitk.GetImageFromArray(grayscale)
test_arr = np.ones((grayscale.shape), dtype='uint8')
ma = sitk.GetImageFromArray(test_arr)
#ma = sitk.GetImageFromArray(np.ones(grayscale.shape),dtype='uint8')


# Store to nrrd:
sitk.WriteImage(im, 'image.nrrd')
sitk.WriteImage(ma, 'mask.nrrd', True)  # enable compression to save disk space

# or extract features:
from radiomics import featureextractor
extractor = featureextractor.RadiomicsFeatureExtractor(r'path/to/params.yml')
features = extractor.execute(im, im, label=1)

glrlm=[]

glrlm.append(float(features['original_glrlm_ShortRunEmphasis']))
glrlm.append(float(features['original_glrlm_LongRunEmphasis']))
glrlm.append(float(features['original_glrlm_GrayLevelNonUniformity']))
glrlm.append(float(features['original_glrlm_RunLengthNonUniformity']))
glrlm.append(float(features['original_glrlm_RunPercentage']))
glrlm.append(float(features['original_glrlm_LowGrayLevelRunEmphasis']))
glrlm.append(float(features['original_glrlm_HighGrayLevelRunEmphasis']))
glrlm.append(float(features['original_glrlm_ShortRunLowGrayLevelEmphasis']))
glrlm.append(float(features['original_glrlm_ShortRunHighGrayLevelEmphasis']))
glrlm.append(float(features['original_glrlm_LongRunLowGrayLevelEmphasis']))
glrlm.append(float(features['original_glrlm_LongRunHighGrayLevelEmphasis']))


df1= pd.DataFrame([glrlm])
df1.columns =['Short_run_emphasis_0', 'Long_run_emphasis_0','Gray_level_nonuniformity_0' ,'Run_length_nonuniformity_0','Run_percentage_0','Low_gray_level_run_emphasis_0','High_gray_level_run_emphasis_0','Short_run_Low_gray_level_emphasis_0','Short_run_High_gray_level_emphasis_0','Long_run_Low_gray_level_emphasis_0','Long_run_High_gray_level_emphasis_0']

df2= pd.DataFrame([glrlm])
df2.columns =['Short_run_emphasis_45', 'Long_run_emphasis_45','Gray_level_nonuniformity_45' ,'Run_length_nonuniformity_45','Run_percentage_45','Low_gray_level_run_emphasis_45','High_gray_level_run_emphasis_45','Short_run_Low_gray_level_emphasis_45','Short_run_High_gray_level_emphasis_45','Long_run_Low_gray_level_emphasis_45','Long_run_High_gray_level_emphasis_45']

df3= pd.DataFrame([glrlm])
df3.columns =['Short_run_emphasis_90', 'Long_run_emphasis_90','Gray_level_nonuniformity_90' ,'Run_length_nonuniformity_90','Run_percentage_90','Low_gray_level_run_emphasis_90','High_gray_level_run_emphasis_90','Short_run_Low_gray_level_emphasis_90','Short_run_High_gray_level_emphasis_90','Long_run_Low_gray_level_emphasis_90','Long_run_High_gray_level_emphasis_90']

df4= pd.DataFrame([glrlm])
df4.columns =['Short_run_emphasis_135', 'Long_run_emphasis_135','Gray_level_nonuniformity_135' ,'Run_length_nonuniformity_135','Run_percentage_135','Low_gray_level_run_emphasis_135','High_gray_level_run_emphasis_135','Short_run_Low_gray_level_emphasis_135','Short_run_High_gray_level_emphasis_135','Long_run_Low_gray_level_emphasis_135','Long_run_High_gray_level_emphasis_135']

frames = [df1, df2, df3, df4]
result_glrlm = pd.concat(frames, axis=1, sort=False)


