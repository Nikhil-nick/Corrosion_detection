from PIL import Image
import numpy as np
import scipy
from scipy import stats
import cv2
import skimage
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt
import pandas as pd

image=cv2.imread(r"C:\Users\Pavan\Desktop\Project\image_1.jpg")
b,g,r=cv2.split(image)
rgb_img=cv2.merge([r,g,b])
plt.imshow(rgb_img)
plt.imshow(image)
red=image[:,:,2]
Mean=list()
Range=list()
Deviation=list()
Entropy=list()
Skewness=list()
Kurtosis=list()
#print(red)


r1=r.mean()
Mean.append(r1)
r2=skimage.measure.shannon_entropy(r,base=10)
Entropy.append(r2)
r3=r.std()
Deviation.append(r3)
r4=np.ptp(r)
Range.append(r4)
t=scipy.stats.skew(r)
#print(b)
r5=t.mean()
Skewness.append(r5)
c=scipy.stats.kurtosis(r,axis=0,fisher=False)
#print(c)
r6=c.mean()
Kurtosis.append(r6)

g1=g.mean()
Mean.append(g1)
g2=skimage.measure.shannon_entropy(g,base=10)
Entropy.append(g2)
g3=g.std()
Deviation.append(g3)
g4=np.ptp(g)
Range.append(g4)
e=scipy.stats.skew(g)
#print(b)
g5=e.mean()
Skewness.append(g5)
f=scipy.stats.kurtosis(g,axis=0,fisher=False)
#print(c)
g6=f.mean()
Kurtosis.append(g6)

b1=b.mean()
Mean.append(b1)
b2=skimage.measure.shannon_entropy(b,base=10)
Entropy.append(b2)
b3=b.std()
Deviation.append(b3)
b4=np.ptp(b)
Range.append(b4)
h=scipy.stats.skew(b)
#print(b)
b5=h.mean()
Skewness.append(b5)
i=scipy.stats.kurtosis(b,axis=0,fisher=False)
#print(c)
b6=i.mean()
Kurtosis.append(b6)

print(Mean)
print(Entropy)
print(Deviation)
print(Range)
print(Skewness)
print(Kurtosis)


#df1= pd.DataFrame(Mean, columns = ['Mean_R', 'Mean_G','Mean_B'])
df1= pd.DataFrame([Mean])
df1.columns =['Mean_R', 'Mean_G','Mean_B']

df2= pd.DataFrame([Deviation])
df2.columns =['Standard_deviation_R', 'Standard_deviation_G','Standard_deviation_B']

df3= pd.DataFrame([Skewness])
df3.columns =['Skewness_R', 'Skewness_G','Skewness_B']

df4= pd.DataFrame([Kurtosis])
df4.columns =['Kurtosis_R', 'Kurtosis_G','Kurtosis_B']

df5= pd.DataFrame([Entropy])
df5.columns =['Entropy_R', 'Entropy_G','Entropy_B']

df6= pd.DataFrame([Range])
df6.columns =['Range_R', 'Range_G','Range_B']

frames = [df1, df2, df3, df4, df5, df6]
result = pd.concat(frames, axis=1, sort=False)
