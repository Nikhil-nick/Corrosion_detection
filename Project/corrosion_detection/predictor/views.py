from django.shortcuts import render
from .smp import smp_values
import cv2
import os
import numpy as np
from django.core.files.storage import default_storage
from django.core.files.storage import FileSystemStorage
from django.core.files.base import ContentFile
from django.conf import settings
from .glcm import glcm_prop
from .glrlm import glrlm_prop
import pandas as pd
import pickle
from PIL import Image

# Create your views here.
def index(request):
    return render(request,'predictor/index.html')

def new_page(request):
    data= ''
    image=''
    new_data = []
    result = []
    result1 = []
    
    if request.POST.get('submit') == 'SUBMIT' :
        myfile = request.FILES['document']
        print("MYFILE is",myfile)
       
        path = default_storage.save(r"C:\Users\Pavan\Desktop\Project\corrosion_detection\predictor\images/img.jpg", ContentFile(myfile.read()))
       
        image=cv2.imread(r"C:\Users\Pavan\Desktop\Project\corrosion_detection\predictor\images\img.jpg")
        stastical_mean = smp_values.mean_properties(image)
        glclm_data = glcm_prop.glclm_properties(image)
        glrlm_data =  glrlm_prop.glrlm_properties(image)
        print(type(stastical_mean))
        print(type(glclm_data))
        print(type(glrlm_data))
        
        frames = [stastical_mean,glclm_data,glrlm_data]
        X_test= pd.concat(frames, axis=1, sort=False)
        model=pickle.load(open(r'C:\Users\Pavan\Desktop\Project\corrosion_detection\predictor\model\model.pkl','rb'))
        Y_pred= model.predict(X_test)
        if(Y_pred==1):
            result="Corroded"
        elif(Y_pred==2):
            result="Not Corroded"
    try: 
        os.remove(r"C:\Users\Pavan\Desktop\Project\corrosion_detection\predictor\images\img.jpg")
    except: pass

    
    context={'data':data, 'result':result}
    return render(request, 'predictor/index.html', context)

    



