# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 18:44:16 2020

@author: User
"""

import numpy
import matplotlib.image as mpimg
from PIL import Image 
  
# Read image 
#img = Image.open('g4g.png')

from radiomics import base, cMatrices

class RadiomicsGLRLM(base.RadiomicsFeaturesBase):
    def __init__(self, inputImage, inputMask, **kwargs):
        super(RadiomicsGLRLM, self).__init__(inputImage, inputMask, **kwargs)

        self.weightingNorm = kwargs.get('weightingNorm', None)  # manhattan, euclidean, infinity

        self.P_glrlm = None
        self.imageArray = self._applyBinning(self.imageArray)
        
    def _initCalculation(self, voxelCoordinates=None):
         
        self.P_glrlm = self._calculateMatrix(voxelCoordinates)
        self._calculateCoefficients()

        self.logger.debug('GLRLM feature class initialized, calculated GLRLM with shape %s', self.P_glrlm.shape)
    def _calculateMatrix(self, voxelCoordinates=None):
        self.logger.debug('Calculating GLRLM matrix in C')
        Ng = self.coefficients['Ng']
        Nr = numpy.max(self.imageArray.shape)
        matrix_args = [
        self.imageArray,
        self.maskArray,
        Ng,
        Nr,
        self.settings.get('force2D', False),
        self.settings.get('force2Ddimension', 0)
        ]
        if self.voxelBased:
            matrix_args += [self.settings.get('kernelRadius', 1), voxelCoordinates]
        P_glrlm, angles = cMatrices.calculate_glrlm(*matrix_args)  # shape (Nvox, Ng, Nr, Na)
        self.logger.debug('Process calculated matrix')
        NgVector = range(1, Ng + 1)  # All possible gray values
        GrayLevels = self.coefficients['grayLevels']  # Gray values present in ROI
        emptyGrayLevels = numpy.array(list(set(NgVector) - set(GrayLevels)))  # Gray values NOT present in ROI

        P_glrlm = numpy.delete(P_glrlm, emptyGrayLevels - 1, 1)
        if self.weightingNorm is not None:
            self.logger.debug('Applying weighting (%s)', self.weightingNorm)
            pixelSpacing = self.inputImage.GetSpacing()[::-1]
            weights = numpy.empty(len(angles))
            for a_idx, a in enumerate(angles):
                if self.weightingNorm == 'infinity':
                    weights[a_idx] = max(numpy.abs(a) * pixelSpacing)
                elif self.weightingNorm == 'euclidean':
                    weights[a_idx] = numpy.sqrt(numpy.sum((numpy.abs(a) * pixelSpacing) ** 2))
                elif self.weightingNorm == 'manhattan':
                    weights[a_idx] = numpy.sum(numpy.abs(a) * pixelSpacing)
                elif self.weightingNorm == 'no_weighting':
                    weights[a_idx] = 1
                else:
                    self.logger.warning('weigthing norm "%s" is unknown, weighting factor is set to 1', self.weightingNorm)
                    weights[a_idx] = 1
            P_glrlm = numpy.sum(P_glrlm * weights[None, None, None, :], 3, keepdims=True)
        Nr = numpy.sum(P_glrlm, (1, 2))
        if P_glrlm.shape[3] > 1:
            emptyAngles = numpy.where(numpy.sum(Nr, 0) == 0)
            if len(emptyAngles[0]) > 0:
                self.logger.debug('Deleting %d empty angles:\n%s', len(emptyAngles[0]), angles[emptyAngles])
                P_glrlm = numpy.delete(P_glrlm, emptyAngles, 3)
                Nr = numpy.delete(Nr, emptyAngles, 1)
            else:
                self.logger.debug('No empty angles')
        Nr[Nr == 0] = numpy.nan  # set sum to numpy.spacing(1) if sum is 0?
        self.coefficients['Nr'] = Nr

        return P_glrlm         
    def _calculateCoefficients(self):
        self.logger.debug('Calculating GLRLM coefficients')

        pr = numpy.sum(self.P_glrlm, 1)  # shape (Nvox, Nr, Na)
        pg = numpy.sum(self.P_glrlm, 2)  # shape (Nvox, Ng, Na)

        ivector = self.coefficients['grayLevels'].astype(float)  # shape (Ng,)
        jvector = numpy.arange(1, self.P_glrlm.shape[2] + 1, dtype=numpy.float64)  # shape (Nr,)

    # Delete columns that run lengths not present in the ROI
        emptyRunLenghts = numpy.where(numpy.sum(pr, (0, 2)) == 0)
        self.P_glrlm = numpy.delete(self.P_glrlm, emptyRunLenghts, 2)
        jvector = numpy.delete(jvector, emptyRunLenghts)
        pr = numpy.delete(pr, emptyRunLenghts, 1)

        self.coefficients['pr'] = pr
        self.coefficients['pg'] = pg
        self.coefficients['ivector'] = ivector
        self.coefficients['jvector'] = jvector
            
    def getShortRunEmphasisFeatureValue(self):
        pr = self.coefficients['pr']
        jvector = self.coefficients['jvector']
        Nr = self.coefficients['Nr']

        sre = numpy.sum((pr / (jvector[None, :, None] ** 2)), 1) / Nr
        return numpy.nanmean(sre, 1)
    def getLongRunEmphasisFeatureValue(self):
        pr = self.coefficients['pr']
        jvector = self.coefficients['jvector']
        Nr = self.coefficients['Nr']

        lre = numpy.sum((pr * (jvector[None, :, None] ** 2)), 1) / Nr
        return numpy.nanmean(lre, 1)
    def getLongRunEmphasisFeatureValue(self):
        pr = self.coefficients['pr']
        jvector = self.coefficients['jvector']
        Nr = self.coefficients['Nr']

        lre = numpy.sum((pr * (jvector[None, :, None] ** 2)), 1) / Nr
        return numpy.nanmean(lre, 1)
    def getGrayLevelNonUniformityFeatureValue(self):
        pg = self.coefficients['pg']
        Nr = self.coefficients['Nr']

        gln = numpy.sum((pg ** 2), 1) / Nr
        return numpy.nanmean(gln, 1)
    def getGrayLevelNonUniformityNormalizedFeatureValue(self):
        pg = self.coefficients['pg']
        Nr = self.coefficients['Nr']

        glnn = numpy.sum(pg ** 2, 1) / (Nr ** 2)
        return numpy.nanmean(glnn, 1)
    def getRunLengthNonUniformityFeatureValue(self):
        pr = self.coefficients['pr']
        Nr = self.coefficients['Nr']

        rln = numpy.sum((pr ** 2), 1) / Nr
        return numpy.nanmean(rln, 1)
    def getRunLengthNonUniformityNormalizedFeatureValue(self):
        pr = self.coefficients['pr']
        Nr = self.coefficients['Nr']

        rlnn = numpy.sum((pr ** 2), 1) / Nr ** 2
        return numpy.nanmean(rlnn, 1)
    def getRunPercentageFeatureValue(self):
        pr = self.coefficients['pr']
        jvector = self.coefficients['jvector']
        Nr = self.coefficients['Nr']

        Np = numpy.sum(pr * jvector[None, :, None], 1)  # shape (Nvox, Na)

        rp = Nr / Np
        return numpy.nanmean(rp, 1)
    def getGrayLevelVarianceFeatureValue(self):
        ivector = self.coefficients['ivector']
        Nr = self.coefficients['Nr']
        pg = self.coefficients['pg'] / Nr[:, None, :]  # divide by Nr to get the normalized matrix

        u_i = numpy.sum(pg * ivector[None, :, None], 1, keepdims=True)
        glv = numpy.sum(pg * (ivector[None, :, None] - u_i) ** 2, 1)
        return numpy.nanmean(glv, 1)
    def getRunVarianceFeatureValue(self):
        jvector = self.coefficients['jvector']
        Nr = self.coefficients['Nr']
        pr = self.coefficients['pr'] / Nr[:, None, :]   # divide by Nr to get the normalized matrix

        u_j = numpy.sum(pr * jvector[None, :, None], 1, keepdims=True)
        rv = numpy.sum(pr * (jvector[None, :, None] - u_j) ** 2, 1)
        return numpy.nanmean(rv, 1)
    def getRunEntropyFeatureValue(self):
        eps = numpy.spacing(1)
        Nr = self.coefficients['Nr']
        p_glrlm = self.P_glrlm / Nr[:, None, None, :]  # divide by Nr to get the normalized matrix

        re = -numpy.sum(p_glrlm * numpy.log2(p_glrlm + eps), (1, 2))
        return numpy.nanmean(re, 1)
    def getLowGrayLevelRunEmphasisFeatureValue(self):
        pg = self.coefficients['pg']
        ivector = self.coefficients['ivector']
        Nr = self.coefficients['Nr']

        lglre = numpy.sum((pg / (ivector[None, :, None] ** 2)), 1) / Nr
        return numpy.nanmean(lglre, 1)
    def getHighGrayLevelRunEmphasisFeatureValue(self):
        pg = self.coefficients['pg']
        ivector = self.coefficients['ivector']
        Nr = self.coefficients['Nr']

        hglre = numpy.sum((pg * (ivector[None, :, None] ** 2)), 1) / Nr
        return numpy.nanmean(hglre, 1)
    def getShortRunLowGrayLevelEmphasisFeatureValue(self):
        ivector = self.coefficients['ivector']
        jvector = self.coefficients['jvector']
        Nr = self.coefficients['Nr']

        srlgle = numpy.sum((self.P_glrlm / ((ivector[None, :, None, None] ** 2) * (jvector[None, None, :, None] ** 2))),
                       (1, 2)) / Nr
        return numpy.nanmean(srlgle, 1)
    def getShortRunHighGrayLevelEmphasisFeatureValue(self):
        ivector = self.coefficients['ivector']
        jvector = self.coefficients['jvector']
        Nr = self.coefficients['Nr']

        srhgle = numpy.sum((self.P_glrlm * (ivector[None, :, None, None] ** 2) / (jvector[None, None, :, None] ** 2)),
                       (1, 2)) / Nr
        return numpy.nanmean(srhgle, 1)
    def getLongRunLowGrayLevelEmphasisFeatureValue(self):
        ivector = self.coefficients['ivector']
        jvector = self.coefficients['jvector']
        Nr = self.coefficients['Nr']

        lrlgle = numpy.sum((self.P_glrlm * (jvector[None, None, :, None] ** 2) / (ivector[None, :, None, None] ** 2)),
                       (1, 2)) / Nr
        return numpy.nanmean(lrlgle, 1)
    def getLongRunHighGrayLevelEmphasisFeatureValue(self):
        ivector = self.coefficients['ivector']
        jvector = self.coefficients['jvector']
        Nr = self.coefficients['Nr']

        lrhgle = numpy.sum((self.P_glrlm * ((jvector[None, None, :, None] ** 2) * (ivector[None, :, None, None] ** 2))),
                       (1, 2)) / Nr
        return numpy.nanmean(lrhgle, 1)

img ='sample.jpg'
test=RadiomicsGLRLM("img","1")
test.getLongRunEmphasisFeatureValue()