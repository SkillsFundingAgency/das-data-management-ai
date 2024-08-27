import xgboost as xgb
import pandas as pd
import numpy as np
import os
import pickle
def main():
    print("HELLO WORLD")
    if(1==1):
        #create a dummy dataframe with same variables as the dummy df used to configure the dummy xgb model
        v1=np.random.randn(100)
        v2=np.random.randn(100)
        v3=np.random.randn(100)
        v4=np.random.randn(100)
        df_t=pd.DataFrame({
            'v1':v1,
            'v2':v2,
            'v3':v3,
            'v4':v4
        })
        print(os.getcwd())
        print(os.system("pip show xgboost"))
    try:
        loaded_model=None    
        with open('dummy_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
    
        print(loaded_model)
    except Exception as e:
        print("Exception: ",e)

    try:
        from azureml.core import Model,Workspace
        # this will fail currently until the workspace IDs are filled
        ws=Workspace.get(
            subscription_id="",
            resource_group="",
            name= ""
        )
        modctr=0

        for model in Model.list(ws):
            if("prod" in model.name.lower()):
                print("iteration:{}".format(modctr))
                print(model.name,model.version)
                if(modctr==0):
                    outpath=model.download(target_dir="./RegisteredModels/",exist_ok=True)
                    print("DOWNLOAD OK")
                    with open(outpath,'rb') as rf:
                        dl_model=pickle.load(rf)
                        print(dl_model)
                    print("DOWNLOADED MODEL AND VERIFIED IT EXISTS")
                modctr+=1
        print("END MODEL READ")
    except Exception as e:
        print("MOD QUERY ERROR: {}".format(e))

 
    return

if __name__=='__main__':
    main()