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
#data.loc[data["ALT"]>150,"ALT"]=151
best_xgb=pickle.load(open(os.path.join(base_dir,'pima1.pickle.dat'), "rb"))
mpl.rcParams['pdf.fonttype'] = 42

shap.initjs()
# 创建 SHAP 解释器
x_test_df = x_test.reset_index(drop=True)
x_test_df = x_test_df.apply(pd.to_numeric, errors='coerce')
explainer = shap.Explainer(best_xgb)
shap_values = explainer(x_test_df)
####这里是需要传的参数 分别为ALT、SMPP、LDH、NLR、Macrolide Treatment、Extensive Lung Consolidation、Duration of Fever days、Peak Fever
###其中SMPP、Macrolide Treatment、Extensive Lung Consolidation改为选择框 1为yes 0为no
# import numpy as np
# # individual_sample = np.array([10, 0, 100.0, 3, 0, 0, 3, 38]).reshape(1, -1)
# individual_sample = np.array([1, 1, 1, 1, 1, 1, 1, 1]).reshape(1, -1)
# predicted_probability = best_xgb.predict_proba(individual_sample)
# shap_values_individual = explainer(individual_sample)
# ###这里是output
# print(f"SMPP Probability: {predicted_probability[0][1]}")  # 这里假设 SMPP 类是正类，索引为1
# shap_values_individual = explainer(individual_sample)
# ###这里是output的图
# # shap.force_plot(explainer.expected_value, shap_values_individual.values[0], individual_sample[0], feature_names=data.columns[0:8])
# shap.force_plot(explainer.expected_value, shap_values_individual.values[0], individual_sample[0], feature_names=data.columns[0:8],show=False, matplotlib=True)
# # plt.show()
# plt.savefig('1.png')