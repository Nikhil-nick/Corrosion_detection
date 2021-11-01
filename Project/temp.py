from PIL import Image
import numpy as np
import scipy
from scipy import stats
import cv2


import matplotlib.pyplot as plt


image=cv2.imread(r"C:\Users\Pavan\Desktop\Project\image_1.jpg")
b,g,r=cv2.split(image)
rgb_img=cv2.merge([r,g,b])
plt.imshow(rgb_img)
plt.imshow(image)
red=image[:,:,2]
print(red)


#data = asarray(image)
#print(type(data))

#print(data.shape)

# create Pillow image
#image2 = Image.fromarray(data)
#print(type(image2))


#print(image2.size)

print(r)
print(g)
print(b)


print(r.var())
print(r.std())
a=np.ptp(r)
print(a)

b=scipy.stats.skew(r)
print(b)
print(b.mean())
c=scipy.stats.kurtosis(r,axis=0,fisher=False)
print(c)
print(c.mean())



