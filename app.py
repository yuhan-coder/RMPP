from flask import Flask,jsonify,render_template,request
from sklearnpre import *
import datetime
import random
import os
import numpy as np
app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    return  render_template('index.html')

@app.route('/api',methods=['GET','POST'])
def api():
    if request.method=='POST':
        print(request.form)
        Duration=request.form.get('Duration','')
        Macrolide=request.form.get('Macrolide','')
        SMPP=request.form.get('SMPP','')
        LDH=request.form.get('LDH','')
        NLR=request.form.get('NLR','')
        ALT=request.form.get('ALT','')
        Peak=request.form.get('Peak','')
        Extensive=request.form.get('Extensive','')
        Duration=float(Duration) if Duration else np.NAN
        Macrolide=float(Macrolide) if Macrolide else np.NAN
        SMPP=float(SMPP) if SMPP else np.NAN
        LDH=float(LDH) if LDH else np.NAN
        NLR=float(NLR) if NLR else np.NAN
        ALT=float(ALT) if ALT else np.NAN
        Peak=float(Peak) if Peak else np.NAN
        Extensive=float(Extensive) if Extensive else np.NAN
        # individual_sample = np.array([10, 0, 100.0, 3, 0, 0, 3, 38]).reshape(1, -1)
        print(type(Duration), Macrolide, SMPP, LDH, NLR, ALT, Peak, Extensive)
        individual_sample = np.array([ ALT,SMPP,LDH,NLR,Macrolide, Extensive,Duration, Peak]).reshape(1, -1)
        # individual_sample = np.array([1, 1, 1, 1, 1, 1, 1, 1]).reshape(1, -1)
        predicted_probability = best_xgb.predict_proba(individual_sample)
        shap_values_individual = explainer(individual_sample)
        ###这里是output
        print(f"Based on feature values, the predicted probability of RMPP is {predicted_probability[0][1]:.2%}")  # 这里假设 SMPP 类是正类，索引为1
        shap_values_individual = explainer(individual_sample)
        ###这里是output的图
        # shap.force_plot(explainer.expected_value, shap_values_individual.values[0], individual_sample[0], feature_names=data.columns[0:8])
        shap.force_plot(explainer.expected_value, shap_values_individual.values[0], individual_sample[0],
                        feature_names=data.columns[0:8], show=False, matplotlib=True)
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),'static/1.png'),dpi=90)
        rand = datetime.datetime.now().strftime('%Y%m%d%H%M%S') + str(random.randint(0, 1000))
        number1=round(predicted_probability[0][1]*100,2)
        return jsonify({'SMPP':str(number1),'rand':rand})

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5001,debug=True)