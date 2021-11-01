from django.apps import AppConfig
from django.conf import settings
import os 
import pickle


class PredictorConfig(AppConfig):
    path=os.path.join(settings.MODELS,'model.pkl')
    with open(path,'rb') as loaded:
        data=pickle.load(loaded)
        

    name = 'predictor'
