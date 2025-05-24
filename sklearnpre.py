import sklearn
from sklearn import datasets
import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve,auc
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from sklearn.metrics import recall_score,confusion_matrix
import pandas as pd
import xgboost as xgb
import pickle
from imblearn.under_sampling import RandomUnderSampler,ClusterCentroids,NearMiss
import shap
import xgboost
import matplotlib.pyplot as plt
import shap
import matplotlib as mpl
import os
base_dir=os.path.dirname(os.path.abspath(__file__))
data=pd.read_csv(os.path.join(base_dir,'RMMP_0527.csv'),index_col=0)
data.iloc[:, [1, 2, 3, 8]] = data.iloc[:, [1, 2, 3, 8]].astype('category')
x_train,x_test,y_train,y_test=model_selection.train_test_split(data.iloc[:, np.r_[0:8]],data.iloc[:,8],test_size=0.25,random_state=25)
x_train,y_train=RandomUnderSampler(random_state=2).fit_resample(x_train, y_train)
best_xgb=pickle.load(open(os.path.join(base_dir,'pima1.pickle.dat'), "rb"))
mpl.rcParams['pdf.fonttype'] = 42

shap.initjs()
# 创建 SHAP 解释器
x_test_df = x_test.reset_index(drop=True)
x_test_df = x_test_df.apply(pd.to_numeric, errors='coerce')
explainer = shap.Explainer(best_xgb)
shap_values = explainer(x_test_df)
