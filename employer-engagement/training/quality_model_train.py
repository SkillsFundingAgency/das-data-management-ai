import pickle
import os
import pandas as pd
import pyarrow.parquet as pq
import joblib
from azureml.core.run import Run
from azureml.core import Dataset, Datastore, Model
from azureml.data.datapath import DataPath
from sklearn.linear_model import LogisticRegression

# split the data into target and predictors
y = quality_sample_data2['completed_commitment']
X = quality_sample_data2[['levy_non_levy','previous_12mon_commitments','apprenticeship_level','apprentice_age','funded_by_levy_transfer',
                'occupation_1','occupation_2','occupation_3','occupation_7','occupation_13','occupation_14','occupation_15',
                'occupation_17','occupation_20','occupation_22','occupation_24','as_months_since_sign_up','log_employees',
                'comp_type_C','comp_type_E','comp_type_F','comp_type_I','comp_type_L','comp_type_P','comp_type_S',
                'comp_type_X','tpr_match','new_company','early_adopter','years_since_tpr_signup']]


# get code from Vemal for train, accuracy and add below and register

# need to group final model score up to account level







#split data into target and independent sets
y_train=df['target']
x_train=df[['dep_var2','dep_var3']]

#build logistic model
clf=LogisticRegression(random_state=0).fit(x_train,y_train)

#print model coefficients
print(clf.coef_)
print(clf.intercept_)

run = Run.get_context()
run.log('quality_model_train.log')

