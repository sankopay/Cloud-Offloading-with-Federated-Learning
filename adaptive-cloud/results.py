from util.time_generation import TimeGeneration
import os


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

def model_class_name == 'ModelSVMSmooth':
        from models.svm_smooth import ModelSVMSmooth
        return ModelSVMSmooth()


X, y = datasets.load(return_X_y=True)
X.shape, y.shape
((150, 4), (150,))