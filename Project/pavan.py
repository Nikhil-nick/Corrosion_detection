
import numpy as np
import cv2
import skimage
from skimage.feature import greycomatrix, greycoprops
import pandas as pd


image=cv2.imread(r"C:\Users\Pavan\Desktop\Project\image_1.jpg")
result=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print(image)
print(result)
print(result.max())
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

entropy=[]
ASM_val=[]
Contrast=[]
Corrleation=[]
new_angle=0
for i in range(4):
    new_angle=angles[i]
    g_e= greycomatrix(result, [1],[new_angle], 256, symmetric=False, normed=True)
        
    entr = skimage.measure.shannon_entropy(g_e)
    correlation = greycoprops(g_e, 'correlation')
    ASM = greycoprops(g_e, 'ASM')
    contrast = greycoprops(g_e, 'contrast')
    
    entropy.append(entr)
    ASM_val.append(ASM[0][0])
    Contrast.append(contrast[0][0])
    Corrleation.append(correlation[0][0])

#asm,contrast,corrleation,entropy 
dummy0=[]
dummy0.append(ASM_val[0])
dummy0.append(Contrast[0])
dummy0.append(Corrleation[0])
dummy0.append(entropy[0])

dummy45=[]
dummy45.append(ASM_val[1])
dummy45.append(Contrast[1])
dummy45.append(Corrleation[1])
dummy45.append(entropy[1])

dummy90=[]
dummy90.append(ASM_val[2])
dummy90.append(Contrast[2])
dummy90.append(Corrleation[2])
dummy90.append(entropy[2])

dummy135=[]
dummy135.append(ASM_val[3])
dummy135.append(Contrast[3])
dummy135.append(Corrleation[3])
dummy135.append(entropy[3])



df1= pd.DataFrame([dummy0])
df1.columns =['ASM_0', 'Contrast_0','Corrleation_0','entropy_0']

df2= pd.DataFrame([dummy45])
df2.columns =['ASM_45', 'Contrast_45','Corrleation_45','entropy_45']

df3= pd.DataFrame([dummy90])
df3.columns =['ASM_90', 'Contrast_90','Corrleation_90','entropy_90']


df4= pd.DataFrame([dummy135])
df4.columns =['ASM_135', 'Contrast_135','Corrleation_135','entropy_135']

frames = [df1, df2, df3, df4]
result_glcm = pd.concat(frames, axis=1, sort=False)



