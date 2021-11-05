# set up libraries
import pandas as pd

#read the data from sql
df = pd.read_sql('SELECT * from test_data',conn)

# create some additional features on the data
def fn_age_grp(row):
    if row['dep_var2']<=30:
        val='A_0_30'
    elif row['dep_var2']<=60:
        val='B_31_60'
    else:
        val='C_61+'
    return val

# create some additional features on the data
def fn_age_grp(row):
    if row['dep_var2']<=30:
        val='A_0_30'
    elif row['dep_var2']<=60:
        val='B_31_60'
    else:
        val='C_61+'
    return val

# apply feature functions
# create age grouping variable
df['grp_age']=df.apply(fn_age_grp,axis=1)

# create sector code from TCN
df['sector']=df['dep_var1'].str.slice(1,3)
