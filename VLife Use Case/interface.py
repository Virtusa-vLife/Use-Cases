import pick
from pick import pick
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns



df=pd.DataFrame()
print("\nEnter the age of the patient")
df.loc[0,'age']=input()

title="Was the admit an emergency?"
options=['Yes','No']
df.loc[0,'emergency']= pick(options, title, min_selection_count=1)[0]

title="Is the patients insurance provider either MEDICARE or MEDICAID?"
options=['Yes','No']
df.loc[0,'insurance']=pick(options, title, min_selection_count=1)[0]

print("\nHow long was the patient in the hostpital?")
df.loc[0,'timeofstay']=input()

print("\nHow many times has the patient been admitted in an emergency ward in the past one year?")
df.loc[0,'total_em_6']=input()

title="Does the patient suffer from any comorbidity along with his/her primary diagnosis?"
options=['Yes','No']
df.loc[0,'comorbid']=pick(options, title, min_selection_count=1)[0]

title="Is the patient's ailment chronic?"
options=['Yes','No']
df.loc[0,'CHRONIC']=pick(options, title, min_selection_count=1)[0]


dict_of_CCI={'Diabetes (uncomplicated)':1,'Diabetes (End Organ Damage)':2,'Liver Disease (mild)':1,'Liver Disease (moderate to severe)':3,'Malignancy (Localized/Leukemia/Lymphoma)':2,'Malignancy (Metastatic)':6,
         'AIDS':6,'Chronic Kidney Disease':2,'Congestive Heart Failure':1,'Myocardial Infarction':1,
         'COPD':1,'Peripheral Vascular Disease':1,'Transient Ischemic Attack':1,'Dementia':1
         ,'Hemiplegia':2,'Connective Tissue Disease':1,'Peptic Ulcer Disease':1}


title="Charlson's Comorbidity Index. Please select those that apply by selecting with space bar and pressing Enter"
options=['Diabetes (uncomplicated)','Diabetes (End Organ Damage)','Liver Disease (mild)','Liver Disease (moderate to severe)','Malignancy (Localized/Leukemia/Lymphoma)','Malignancy (Metastatic)',
         'AIDS','Chronic Kidney Disease','Congestive Heart Failure','Myocardial Infarction',
         'COPD','Peripheral Vascular Disease','Transient Ischemic Attack','Dementia'
         ,'Hemiplegia','Connective Tissue Disease','Peptic Ulcer Disease']

CCI=pick(options, title, multi_select=True, min_selection_count=0)

icd9_code=0

for x in CCI:
    icd9_code=icd9_code+dict_of_CCI[x[0]]

df.loc[0,'icd9_code']=icd9_code



df=df[['emergency',
 'insurance',
 'timeofstay',
 'age',
 'total_em_6',
 'comorbid',
 'icd9_code',
 'CHRONIC']]

def convert_to_binary(x):
    if x=='Yes':
        return 1
    elif x=='No':
        return 0
    else:
        return x

df=df.applymap(convert_to_binary)
df=df.astype(float)


df.loc[0,'age']=df.loc[0,'age']/89
df.loc[0,'icd9_code']=(df.loc[0,'icd9_code'])/56
df.loc[0,'timeofstay']=(df.loc[0,'timeofstay'])/294
df.loc[0,'total_em_6']=(df.loc[0,'total_em_6'])/23




print(df)
LogReg = pickle.load(open('interface.sav', 'rb'))


l=list(LogReg.predict_proba(df.loc[[0]]).ravel().round(2))



sns.set()
plt.figure(figsize=(15,10))
ax=sns.barplot(y=l,x=['No Readmission','Readmission Within One Month'])
plt.ylim(0,1)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
    p.get_height()+0.02,
    '{:1.1f}%'.format(height*100),
    ha="center",color='black',fontsize=30)


df=df.astype(float)


print(df)

a=[0.217, 0.251, 3.367, 0.164, 11.104, 0.325, 1.049, 0.074]
b=list(df.loc[[0]].values.ravel())

print(a)
print('\n',b)

factors=[0,0,0,0,0,0,0,0]
for x in range(0,8):
    factors[x]=(a[x])*(b[x])

s = sum(factors); norm = [float(i)/s for i in factors]

sns.set()
plt.figure(figsize=(15,10))
ax=sns.barplot(y=norm,x=['Emergency','Insurance','Time Of Stay','Age','Past Emergencies','Comorbidites','CCI','Chronic'])
plt.ylim(0,1)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
    p.get_height()+0.02,
    '{:1.1f}%'.format(height*100),
    ha="center",color='black',fontsize=30)

plt.show()