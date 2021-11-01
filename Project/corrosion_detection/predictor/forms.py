# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:15:23 2020

@author: Pavan
"""

class imguploadform(forms.ModelForm):
    class Meta:
        model = imgupload
        fields = ['image']