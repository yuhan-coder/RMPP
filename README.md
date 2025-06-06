# README for the RMPP Prediction Tool

This README provides an overview of the refractory mycoplasma pneumoniae pneumonia(RMPP) prediction tool, its purpose, and how to use it effectively.

## Overview

The RMPP Prediction Tool is a web-based application designed to predict the probability of a patient developing RMPP. The tool takes into account various clinical variables and uses a machine learning model to provide a probabilistic prediction. This tool is intended to aid clinicians in making informed decisions about patient care.

When the actual values of the 8 required features are input into the model, this application will automatically predict the risk of RMPP for an individual child. Additionally, it will display a SHAP force plot for the child, indicating the features contributing to the RMPP decision: the blue features on the right push the prediction towards non-RMPP, while the red features on the left push the prediction towards RMPP.

---
![image text](https://github.com/yuhan-coder/RMPP/blob/main/static/IMG1.png)


## Prerequisites

Before running the RMPP Prediction Tool, ensure that you have the following installed on your system:
- Python 3.5 or later

## Download the Project

Clone the repository to your local machine.

## Running the Application

To start the application, navigate to the `RMPP-main` directory and run `app.py`:

```bash
cd ./RMPP-main/
## Running the Application
python ./app.py

# After starting the application using the previous steps, you should see output similar to:
# Running on all addresses (0.0.0.0)
# Running on http://xxx.0.0.1:5000 your port
# Running on http://xxx.xxx.53.212:5000 your port
# You can choose any of the above URLs and paste it into your web browser to access the RMPP Prediction Tool.

```
Once you open the URL, you can run this app multiple times to guide your predictions on the risk of RMPP for the child.


## Clinical Variables

The tool requires the following input variables:

- **Duration of Fever (days)**: Number of days the patient has had a fever.
- **SMPP**: Whether the patient has Severe Mycoplasma Pneumoniae Pneumonia (Yes=1/No=0).
- **NLR**: Neutrophil-to-Lymphocyte Ratio.
- **Peak Fever**: Highest recorded body temperature during the fever (℃).
- **Macrolide Treatment**: Whether the patient has received macrolide treatment (Yes=1/No=0).
- **LDH**: Lactate Dehydrogenase levels (U/L).
- **ALT**: Alanine Aminotransferase levels (U/L).
- **Extensive Lung Consolidation**: Presence of extensive lung consolidation (Yes=1/No=0).

## How to Use

### Input Clinical Data

1. Fill in the relevant fields with the patient's clinical data.
2. Select appropriate options from dropdown menus where applicable (e.g., Yes/No options).

### Predict

1. Click the "Predict" button to generate the prediction.
2. The predicted probability of RMPP will be displayed below the input fields.

### Interpret Results

1. Review the predicted probability.
2. Examine the SHAP values visualization to understand the contribution of each feature to the prediction.

## Citation  
If you use this data, please cite:  
> **Predicting and interpreting key features of refractory Mycoplasma pneumoniae pneumonia using multiple machine learning methods.** Sci Rep 15, 18029 (2025). https://doi.org/10.1038/s41598-025-02962-4

## Contact  
For questions or collaborations, please contact:  
📧 Email:jiangyuhan@tmu.edu.cn
