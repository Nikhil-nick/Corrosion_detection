# -*- coding: utf-8 -*-
"""
Created on Mon May 11 19:11:52 2020

@author: Pavan
"""

import os

import SimpleITK as sitk
import six

from radiomics import featureextractor, getTestCase

dataDir = '/path/to/pyradiomics'

imageName, maskName = getTestCase('brain1', dataDir)

params = os.path.join(dataDir, "examples", "exampleSettings", "Params.yaml")

extractor = featureextractor.RadiomicsFeatureExtractor(params)

result = extractor.execute(imageName, maskName)
for key, val in six.iteritems(result):
  print("\t%s: %s" %(key, val))