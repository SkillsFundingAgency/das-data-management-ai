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
#X = quality_sample_data2[['levy_non_levy','previous_12mon_commitments','apprenticeship_level','apprentice_age','funded_by_levy_transfer',
#                'occupation_1','occupation_2','occupation_3','occupation_7','occupation_13','occupation_14','occupation_15',
#                'occupation_17','occupation_20','occupation_22','occupation_24','as_months_since_sign_up','log_employees',
#                'comp_type_C','comp_type_E','comp_type_F','comp_type_I','comp_type_L','comp_type_P','comp_type_S',
#                'comp_type_X','tpr_match','new_company','early_adopter','years_since_tpr_signup']]

############################# Add back in ################################

X = quality_sample_data2[['levy_non_levy','previous_12mon_commitments','apprenticeship_level','apprentice_age','funded_by_levy_transfer', \
                'occupation_1','occupation_2','occupation_3','occupation_7','occupation_13','occupation_14','occupation_15', \
                'occupation_17','occupation_20','occupation_22','occupation_24','as_months_since_sign_up','log_employees', \
                'comp_type_C','comp_type_I', \
                'comp_type_X','tpr_match','new_company','early_adopter','years_since_tpr_signup']]


# get code from Vemal for train, accuracy and add below and register

# need to group final model score up to account level

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_train)

shap.summary_plot(shap_values, X_train)


run = Run.get_context()
run.log('quality_model_train_log')

# Save the trained model in the outputs folder
print("Saving model...")
os.makedirs('outputs', exist_ok=True)
model_file = os.path.join('outputs', 'quality_model.pkl')
joblib.dump(value=xgb_model, filename=model_file)

# Register the model to pick up in scoring
print('Registering model...')
Model.register(workspace=aml_workspace, \
               model_path = model_file, \
               model_name = 'quality_model')