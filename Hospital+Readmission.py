
# coding: utf-8

# # Import Requisite Libraries
# Tables in the form of CSVs in root file

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle

from scipy import stats
from pandas.tools.plotting import scatter_matrix
from IPython.display import display

pd.options.display.max_columns = None

tables_req = ['admissions','caregivers','cptevents','patients','prescriptions','d_cpt','d_icd_diagnoses','d_icd_procedures','d_items','d_labitems','diagnoses_icd','drgcodes','icustays','inputevents_cv','inputevents_mv','labevents','microbiologyevents','patients','prescriptions','procedureevents_mv','procedures_icd','services','transfers']

print(len(tables_req))


# In[2]:

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))


# from tkinter import *
# 
# master = Tk()
# 
# variable = StringVar(master)
# variable.set("one") # default value
# 
# w = OptionMenu(master, variable, "one", "two", "three")
# w.pack()
# 
# mainloop()

# from tkinter import *
# 
# 
# # Here, we are creating our class, Window, and inheriting from the Frame
# # class. Frame is a class from the tkinter module. (see Lib/tkinter/__init__)
# class Window(Frame):
# 
#     # Define settings upon initialization. Here you can specify
#     def __init__(self, master=None):
#         
#         # parameters that you want to send through the Frame class. 
#         Frame.__init__(self, master)   
# 
#         #reference to the master widget, which is the tk window                 
#         self.master = master
# 
#         #with that, we want to then run init_window, which doesn't yet exist
#         self.init_window()
# 
#     #Creation of init_window
#     def init_window(self):
# 
#         # changing the title of our master widget      
#         self.master.title("GUI")
# 
#         # allowing the widget to take the full space of the root window
#         self.pack(fill=BOTH, expand=1)
# 
#         # creating a button instance
#         quitButton = Button(self, text="Exit",command=self.client_exit)
# 
#         # placing the button on my window
#         quitButton.place(x=0, y=0)
# 
#        
# 
#     def client_exit(self):
#         exit()
# 
# # root window created. Here, that would be the only window, but
# # you can later have windows within windows.
# root = Tk()
# 
# root.geometry("400x300")
# 
# #creation of an instance
# app = Window(root)
# 
# #mainloop 
# root.mainloop()  

# In[3]:

def race_reassign(x):
    if 'BLACK' in x:
        return 'BLACK'
    elif 'ASIAN' in x:
        return 'ASIAN'
    elif 'HISPANIC' in x:
        return 'HISPANIC'
    elif 'WHITE' in x:
        return'WHITE'
    else:
        return x   
        


# In[4]:

def decreasing_normalized_bar(x):
    plt.figure(figsize=(15,10))
    a=df.groupby([x,'readmitted']).size().unstack()
    c=a[0]+a[1]
    a=a.div(a.sum(axis=1), axis=0)
    a=a.sort_values(1,ascending=False)
    a['number']=c
    
    sns.barplot(x=a.index, y=a[1])
    plt.ylabel('Percentage Readmiited')
    plt.xlabel(x)
    plt.ylim(0,1)
    plt.xticks(rotation=+90)
    plt.figure(figsize=(15,10))
    plt.savefig('{}.png'.format(x))
    
    return a
    


# In[ ]:




# In[5]:

def bar(a):
    plt.figure(figsize=(15,10))
    sns.countplot(x=a, data=df, hue='readmitted')
    plt.xticks(rotation=+90)


# In[6]:

def religion_reassign(x):
    if x in ['7TH DAY ADVENTIST',
 'BAPTIST',
 'CATHOLIC',
 'CHRISTIAN SCIENTIST',
 'EPISCOPALIAN',
 'GREEK ORTHODOX',
 "JEHOVAH'S WITNESS",
 'METHODIST',
 'PROTESTANT QUAKER',
 'ROMANIAN EAST. ORTH',
 'UNITARIAN-UNIVERSALIST']:
        return 'CHRISTIAN'
    else:
        return x


# In[7]:

def race_reassign(x):
    if 'BLACK' in x:
        return 'BLACK'
    elif 'ASIAN' in x:
        return 'ASIAN'
    elif 'HISPANIC' in x:
        return 'HISPANIC'
    elif 'WHITE' in x:
        return'WHITE'
    else:
        return x   
        
#df['ethnicity']=df['ethnicity'].apply(race_reassign)


# In[8]:

def pie_distr(x):
    fig = plt.figure(figsize=(15,8))
    i = 0
    for y in filter(None,df[x].dropna().unique()):
        fig.add_subplot(6, 3, i+1)
        plt.title(y)
        df.readmitted[df[x] == y].value_counts().plot(kind='pie',autopct='%1.0f%%')
        print((df[x]==y).sum())
        i += 1


# In[9]:

def ordered_hist(x):
    plt.figure(figsize=(15,10))
    sns.barplot(df[x].value_counts().index, df[x].value_counts().values)
    plt.xticks(rotation=+90)


# In[10]:

def percent_hist(x):
    y=df.groupby(x)['readmitted'].mean()
    return y.plot.bar()


# In[11]:

def chi2_table(series1, series2, to_csv = False, csv_name = None, 
                prop= False):
    
    if type(series1) != list:
        crosstab = pd.crosstab(series1, series2)
        crosstab2 = pd.crosstab(series1, series2, margins= True)
        crosstab_proprow = round(crosstab2.div(crosstab2.iloc[:,-1], axis=0).mul(100, axis=0), 2)
        crosstab_propcol = round(crosstab2.div(crosstab2.iloc[-1,:], axis=1).mul(100, axis=1), 2)
        chi2, p, dof, expected = stats.chi2_contingency(crosstab)
        
        if prop == False:
            print("\n",
          f"Chi-Square test between " + series1.name + " and " + series2.name,
          "\n", "\n",
          crosstab2,
          "\n", "\n",
          f"Pearson Chi2({dof})= {chi2:.4f} p-value= {p:.4f}")
            
            if to_csv == True:
                if csv_name == None:
                    csv_name = f"{series2.name}.csv"
                                             
                file = open(csv_name, 'a')
                file.write(f"{crosstab2.columns.name}\n")
                file.close()
                crosstab2.to_csv(csv_name, header= True, mode= 'a')
                file = open(csv_name, 'a')
                file.write(f"Pearson Chi2({dof})= {chi2:.4f} p-value= {p:.4f}")
                file.write("\n")
                file.close()              
                
        if prop == 'Row':
            print("\n",
          f"Chi-Square test between " + series1.name + " and " + series2.name,
          "\n", "\n",
          crosstab_proprow,
          "\n", "\n",
          f"Pearson Chi2({dof})= {chi2:.4f} p-value= {p:.4f}")
            
            if to_csv == True:
                if csv_name == None:
                    csv_name = f"{series2.name}.csv"
                
                file = open(csv_name, 'a')
                file.write(f"{crosstab_proprow.columns.name}\n")
                file.close()
                crosstab_proprow.to_csv(csv_name, header= True, mode= 'a')
                file = open(csv_name, 'a')
                file.write(f"Pearson Chi2({dof})= {chi2:.4f} p-value= {p:.4f}")
                file.write("\n")
                file.close()

        if prop == 'Col':
            print("\n",
          f"Chi-Square test between " + series1.name + " and " + series2.name,
          "\n", "\n",
          crosstab_propcol,
          "\n", "\n",
          f"Pearson Chi2({dof})= {chi2:.4f} p-value= {p:.4f}")

            if to_csv == True:
                if csv_name == None:
                    csv_name = f"{series2.name}.csv"
                    
                file = open(csv_name, 'a')
                file.write(f"{crosstab_propcol.columns.name}\n")
                file.close()
                crosstab_propcol.to_csv(csv_name, header= True, mode= 'a')
                file = open(csv_name, 'a')
                file.write(f"Pearson Chi2({dof})= {chi2:.4f} p-value= {p:.4f}")
                file.write("\n")
                file.close()

    elif type(series1) == list and type(series2) == list:
        for entry2 in series2:
            for entry1 in series1:
                crosstab = pd.crosstab(entry1, entry2)
                crosstab2 = pd.crosstab(entry1, entry2, margins= True)
                crosstab_proprow = round(crosstab2.div(crosstab2.iloc[:,-1], axis=0).mul(100, axis=0), 2)
                crosstab_propcol = round(crosstab2.div(crosstab2.iloc[-1,:], axis=1).mul(100, axis=1), 2)
                chi2, p, dof, expected = stats.chi2_contingency(crosstab)
                
                if prop == False:
            
                    print("\n",
          f"Chi-Square test between " + entry1.name + " and " + entry2.name,
          "\n", "\n",
          crosstab2,
          "\n", "\n",
          f"Pearson Chi2({dof})= {chi2:.4f} p-value= {p:.4f}")
            
                    if to_csv == True:
                        
                        file = open("%s.csv" %(entry2.name), 'a')
                        file.write(f"{crosstab2.columns.name}\n")
                        file.close()
                        crosstab2.to_csv("%s.csv" %(entry2.name), header= True, mode= 'a')
                        file = open("%s.csv" %(entry2.name), 'a')
                        file.write(f"Pearson Chi2({dof})= {chi2:.4f} p-value= {p:.4f}")
                        file.write("\n")
                        file.close()                        

                if prop == 'Row':
            
                    print("\n",
          f"Chi-Square test between " + entry1.name + " and " + entry2.name,
          "\n", "\n",
          crosstab_proprow,
          "\n", "\n",
          f"Pearson Chi2({dof})= {chi2:.4f} p-value= {p:.4f}")
            
                    if to_csv == True:
                        file = open("%s.csv" %(entry2.name), 'a')
                        file.write(f"{crosstab_proprow.columns.name}\n")
                        file.close()
                        crosstab_proprow.to_csv("%s.csv" %(entry2.name), header= True, mode= 'a')
                        file = open("%s.csv" %(entry2.name), 'a')
                        file.write(f"Pearson Chi2({dof})= {chi2:.4f} p-value= {p:.4f}")
                        file.write("\n")
                        file.close()
                    
                if prop == 'Col':
            
                    print("\n",
          f"Chi-Square test between " + entry1.name + " and " + entry2.name,
          "\n", "\n",
          crosstab_propcol,
          "\n", "\n",
          f"Pearson Chi2({dof})= {chi2:.4f} p-value= {p:.4f}")
            
                    if to_csv == True:
                        file = open("%s.csv" %(entry2.name), 'a')
                        file.write(f"{crosstab_propcol.columns.name}\n")
                        file.close()
                        crosstab_propcol.to_csv("%s.csv" %(entry2.name), header= True, mode= 'a')
                        file = open("%s.csv" %(entry2.name), 'a')
                        file.write(f"Pearson Chi2({dof})= {chi2:.4f} p-value= {p:.4f}")
                        file.write("\n")
                        file.close()


    elif type(series1) == list:
        for entry in series1:
            crosstab = pd.crosstab(entry, series2)
            crosstab2 = pd.crosstab(entry, series2, margins= True)
            crosstab_proprow = round(crosstab2.div(crosstab2.iloc[:,-1], axis=0).mul(100, axis=0), 2)
            crosstab_propcol = round(crosstab2.div(crosstab2.iloc[-1,:], axis=1).mul(100, axis=1), 2)
            chi2, p, dof, expected = stats.chi2_contingency(crosstab)
            
            if prop == False:
                print("\n",
          f"Chi-Square test between " + entry.name + " and " + series2.name,
          "\n", "\n",
          crosstab2,
          "\n", "\n",
          f"Pearson Chi2({dof})= {chi2:.4f} p-value= {p:.4f}")
            
                if to_csv == True:
                    file = open("%s.csv" %(series2.name), 'a')
                    file.write(f"{crosstab2.columns.name}\n")
                    file.close()
                    crosstab2.to_csv("%s.csv" %(series2.name), header= True, mode= 'a')
                    file = open("%s.csv" %(series2.name), 'a')
                    file.write(f"Pearson Chi2({dof})= {chi2:.4f} p-value= {p:.4f}")
                    file.write("\n")
                    file.close()

            if prop == 'Row':
                print("\n",
          f"Chi-Square test between " + entry.name + " and " + series2.name,
          "\n", "\n",
          crosstab_proprow,
          "\n", "\n",
          f"Pearson Chi2({dof})= {chi2:.4f} p-value= {p:.4f}")
            
                if to_csv == True:
                    file = open("%s.csv" %(series2.name), 'a')
                    file.write(f"{crosstab_proprow.columns.name}\n")
                    file.close()
                    crosstab_proprow.to_csv("%s.csv" %(series2.name), header= True, mode= 'a')
                    file = open("%s.csv" %(series2.name), 'a')
                    file.write(f"Pearson Chi2({dof})= {chi2:.4f} p-value= {p:.4f}")
                    file.write("\n")
                    file.close()

            if prop == 'Col':
                print("\n",
          f"Chi-Square test between " + entry.name + " and " + series2.name,
          "\n", "\n",
          crosstab_propcol,
          "\n", "\n",
          f"Pearson Chi2({dof})= {chi2:.4f} p-value= {p:.4f}")
            
                if to_csv == True:
                    file = open("%s.csv" %(series2.name), 'a')
                    file.write(f"{crosstab_propcol.columns.name}\n")
                    file.close()
                    crosstab_propcol.to_csv("%s.csv" %(series2.name), header= True, mode= 'a')
                    file = open("%s.csv" %(series2.name), 'a')
                    file.write(f"Pearson Chi2({dof})= {chi2:.4f} p-value= {p:.4f}")
                    file.write("\n")
                    file.close()


# ### Store the dataframes in a dictionary where the dataframe is under a key of the same name

# 
# #d={}
# #for x in tables_req:
#         
# #        dataframe=pd.read_csv('/tables/{}.csv'.format(x))
# #        d["{}".format(x)]=dataframe
# 
# #d['d_icd_diagnoses'][d['d_icd_diagnoses']['long_title'].str.contains("n")]

# In[5]:

def capitalize_after_hyphen(x):
    a=list(x)
    a[p.index('-')+1]=a[p.index('-')+1].capitalize()
    x=''.join(a)
    return ''.join(a)

import pandas as pd
import requests  
#l=['patients','admdissions','diagnoses','drg-codes','icu-stays','procedures','prescriptions','d-icd-diagnoses','d-icd-procedures']
url1="http://ec2-54-88-151-77.compute-1.amazonaws.com:3001/v1/patients?limit=50000&offset=0"
url2="http://ec2-54-88-151-77.compute-1.amazonaws.com:3001/v1/admissions?limit=50000&offset=0"
url3="http://ec2-54-88-151-77.compute-1.amazonaws.com:3001/v1/diagnoses?limit=50000&offset=0"
url4="http://ec2-54-88-151-77.compute-1.amazonaws.com:3001/v1/drg-codes?limit=50000&offset=0"
url5="http://ec2-54-88-151-77.compute-1.amazonaws.com:3001/v1/icu-stays?limit=50000&offset=0"
url6="http://ec2-54-88-151-77.compute-1.amazonaws.com:3001/v1/procedures?limit=50000&offset=0"
url7="http://ec2-54-88-151-77.compute-1.amazonaws.com:3001/v1/prescriptions?limit=50000&offset=0"
url8="http://ec2-54-88-151-77.compute-1.amazonaws.com:3001/v1/d-icd-diagnoses?limit=50000&offset=0"
url9="http://ec2-54-88-151-77.compute-1.amazonaws.com:3001/v1/d-icd-procedures?limit=50000&offset=0" 
d={}
url=[url1,url2,url3,url4,url5,url6,url7,url8,url9]

for x in url:  
    p = x[(x.index('v1/')+len('v1/')):x.index('?limit')]
    try:
        p=capitalize_after_hyphen(p)
    except:
        pass
    try:
        p=p[:p.index('-')]+p[p.index('-')+1:]
    except:
        pass
    
    try:
        p=capitalize_after_hyphen(p)
    except:
        pass
    try:
        p=p[:p.index('-')]+p[p.index('-')+1:]
    except:
        pass
    
    
    print(p)
    
    d['{}'.format(p)]=pd.DataFrame(requests.get(x).json()['{}'.format(p)])
    d['{}'.format(p)].head(1000).to_csv('{}.csv'.format(p),encoding='utf-8', index=False)


# 
# ### list of CSVs

# #### Drop Row ID, immaterial

# In[9]:

d['admissions'].loc[:1000,:]


# In[13]:

#drop row_id as it is useless
for x in d:
    d[x]=d[x].drop(['row_id'],axis=1)


# #### Remove patients who died as they do not contribute meaningfully to readmission data

# In[14]:

d.keys()


# In[15]:

d['patients']=d['patients'][d['patients'].expire_flag==0]
d['patients']=d['patients'].drop(['dod','dod_hosp','dod_ssn','expire_flag'],axis=1)                                


# #### 9011 patients did not die. The ones who did do not contribute meaningfully to patient readmission predictions. Hence they are dropped. Language is dropped due to high null values. 

# In[16]:

#drop all columns where patient died

d['admissions'] = d['admissions'][d['admissions']['deathtime'].isnull()]

#drop language, marital status, edregtime, edouttime, hospitalexpireflag, haschartseventsdata

d['admissions']=d['admissions'].drop(['language','edregtime','edouttime','hospital_expire_flag','has_chartevents_data'],axis=1)

#rearrange all tables by subject_id

for x in d:   
    if 'subject_id' in list(d[x]):
        d[x]=d[x].sort_values('subject_id')
        d[x]=d[x].reset_index(drop=True)

#create a time in hospital column

d['patients']['dob']=pd.to_datetime(d['patients']['dob'])
d['admissions']['dischtime'] = pd.to_datetime(d['admissions']['dischtime'])
d['admissions']['admittime'] = pd.to_datetime(d['admissions']['admittime'])

#create a new column measuring time of stay in hospital

d['admissions']['time_of_stay']=d['admissions']['dischtime']-d['admissions']['admittime']

#remove time of death, all are null anyway

d['admissions']=d['admissions'].drop(['deathtime'],axis=1)


# In[17]:

d['admissions']


# In[18]:

d['admissions']= d['admissions'].sort_values(['subject_id','admittime'])
d['admissions']= d['admissions'].sort_values(['subject_id','dischtime'])
d['admissions']=d['admissions'].reset_index(drop=True)

#Calculate time between readmission
for x in range(1,d['admissions'].shape[0]):
    if d['admissions'].loc[x,'subject_id']==d['admissions'].loc[x-1,'subject_id']:
        
        d['admissions'].loc[x,'time_between_readmission']=d['admissions'].loc[x,'admittime']-d['admissions'].loc[x-1,'dischtime']

#Number of readmits = no. of occurences of subject ID till that point
d['admissions']['number_of_readmits']=d['admissions'].groupby('subject_id').cumcount()


# In[19]:

d['admissions']


# In[20]:

#merge patients, keep left columns on as merging all would result in a large number of nulls
df=pd.merge(d['admissions'],d['patients'], on='subject_id',how='outer')


# In[21]:

#null hadm's have a high missing value count in other columns too due to possibly poor collection of data. Hence, drop
#df=df[df['hadm_id'].notnull()]

#Fill with most common religion
df['religion']=df['religion'].fillna(df['religion'].mode())


# #Unnecessary 
# del d['d_cpt']
# del d['d_items']
# del d['caregivers']
# del d['cptevents']

# In[22]:

df


# In[23]:

#calculate age
df['age']=df['admittime']-df['dob']

#convert to days
df['age']=df['age'].dt.days

#Convert to years
df['age']=df['age']/365

#Erronous data
#df=df.drop(df[df['age']<0].index)


# In[24]:

df.isnull().sum()


# In[25]:

#If diagnosis is newborn, age must be 0
for x in range(df.shape[0]):
    if df.iloc[x,10]=='NEWBORN':
        df.iloc[x,15]=0.0


# plt.figure(figsize=(10,6))
# sns.countplot(df['age'])
# plt.show()
# 
# df[df['age']>-1]['age'].hist()

# In[26]:

#drop date of birth as only age has any meaningful value
df=df.drop('dob',axis=1)

#same patient will have same age, approximated by the fact that the patient will be admitted to the hospital around the same time
df['age']=df.groupby(["subject_id"]).age.apply(lambda x: x.fillna(x))

#Intuitively, discharge location should not be a factor as it wont impact the doctor's decision of discharging the patient
df=df.drop('discharge_location',axis=1)


# In[27]:

d['prescriptions'].drop(['hadm_id','drug_name_poe','drug_name_generic','drug_type','startdate','enddate','formulary_drug_cd','gsn','ndc','prod_strength','dose_val_rx','dose_unit_rx','form_val_disp','form_unit_disp','route'],axis=1,inplace=True)


# In[28]:

#Fill in missing age values by a random normal with 1437 values spread identical to our age histogram
df['age'][df['age'].isnull()]=np.random.normal(df['age'].mean(),df['age'].std(),len(df['age'].isnull()))


# In[29]:

df['age']=df['age'].astype(int)


# In[30]:

#Fill in Catholic as it is the most common by far
df['religion'].fillna('CATHOLIC',inplace=True)
#Convert to days
df['time_between_readmission']=df['time_between_readmission'].dt.days


# In[31]:

#rows_to_columns coverts those tables that contain multiple patient records of the same field into a horizontal vector of binary
#values indicating the presence/absence of that particular feature
def rows_to_columns(a,m):
    b=pd.DataFrame(columns=list(a))
    b['subject_id']=a['subject_id'].unique()
    b.set_index('subject_id',inplace=True)
    for x in range(0,a.shape[0]):
        y=a.loc[x,'subject_id']
        z=a.loc[x,m]
        b.loc[y,z]=1
    b.drop([m],axis=1,inplace=True)
    b=b.reset_index()
    b.fillna(0,inplace=True)
    return b


rows_to_columns(d['prescriptions'],'drug')

#Create binary columns for all drugs
df=pd.merge(df,rows_to_columns(d['prescriptions'],'drug'),on='subject_id',how='left')
#df=pd.merge(df,d['prescriptions'],on='subject_id',how='left')
df.drop(['icustay_id'],axis=1,inplace=True)

#assign 0,1 depending on whether patient got readmitted

df['readmitted'] = np.where(df['number_of_readmits']>0.0, 1, 0)
df['readmitted']=df['readmitted'].shift(-1)

#Replace NaN w/0
df.iloc[:, 15:-1]=df.iloc[:, 15:-1].fillna(0)
df.iloc[:, 15:-1]=df.iloc[:, 15:-1].applymap(np.int64)


# In[32]:

#Fill missing religion values with most common religion value
df['religion']=df.groupby('ethnicity').religion.apply(lambda x: x.fillna(x.mode()))

#drop rows with no hadmid
df=df[df['hadm_id'].notnull()]

#drop empty readmitted columns, as it is our dependent variable
#df=df.drop(df[df['number_of_readmits'].isnull()].index)

#Drop NaN columns
df = df.loc[:, df.columns.notnull()]


# ## Does taking a particular drug cause readmissions?
# 
# 
# 
# Here, we conduct a z test.
# 
# 
# $p_1 = $ proportion of people who took the drug and got readmitted
# 
# $p_2 = $ proportion of people who didnt take the drug and get readmitted
# 
# $p_0 = $ proportion of people who got readmitted in the entire sample
# 
#  $ H_0 : p_1 = p_2$
# 
# $i.e$ proportion of patients who got readmitted after taking the drug vs those who got readmitted without taking the drug are identical.
# 
# 
# $ H_1 : p_1 > p_2 $ 
# $i.e$ proportion of patients readmitted increase after taking a drug
# 
# For a $95$% confidence interval, or $\alpha=0.05$, for one tailed $z_{critical}=1.645$
# 
# Our $z$ value is $z=\frac{p_1-p_2}{\sqrt{p_0(1-p_0)(1/n_1+1/n_2)}}$
# 
# The following function returns the p value of the error in assuming the alternate.
# 
# 
# 

# In[33]:

import scipy.stats as st
def z_score(x):    
    q=df.groupby([x,'readmitted']).size().unstack()
    if q[1][1]==np.nan:
        q[1][1]=0
    p1=(q.loc[1,1]/(q.loc[1,:].sum())).round(3)
    p2=(q.loc[0,1]/(q.loc[0,:].sum())).round(3)
    p0=((q.loc[1,1]+q.loc[0,1])/(q.loc[1,1]+q.loc[0,1]+q.loc[1,0]+q.loc[0,0])).round(3)
    n1=q.loc[0,:].sum()
    n2=q.loc[1,:].sum()
    z=(p1-p2)/((((1/n1)+(1/n2))*p0*(1-p0))**0.5)
    
        
    
    return (1-st.norm.cdf(z)).round(6)


# In[34]:

l=df.iloc[:,15:-1].columns

Meds=pd.DataFrame(index=l)

for x in l:
    try:
        y=df.groupby([x,'readmitted']).size().unstack()
    
        z=y.iloc[1,0]+y.iloc[1,1]
        y=y.div(y.sum(axis=1),axis=0)

        Meds.loc[x,'% readmitted w/drug']=y[1][1].round(2)
        Meds.loc[x,'% readmitted w/o drug']=y[1][0].round(2)
        Meds.loc[x,'Number Who Took Drug']=z.astype(int)
        Meds.loc[x,'p value']=z_score(x)
    except:
        pass
    


# In[35]:

def matrix(x):
    return df.groupby([x,'readmitted']).size().unstack()


# In[36]:

Meds=Meds.sort_values('% readmitted w/drug',ascending=False)


# In[37]:

Meds['Number Who Took Drug'].sort_values()


# In[45]:

Meds


# 
# 
# As is evident, the p values in general are alarmingly low but this could also be due to the incredibly small dataset.
#   
# Thus, we include the count of the medicines with a greater than 80% readmission rate.
# 
# 
# 
# 

# In[39]:

#df.rename(columns = {'Lidocaine 1%':'Lidocaine'}, inplace = True)
#df.rename(columns = {'Albumin 25% (12.5g / 50mL)':'Albumin'}, inplace = True)
#df.rename(columns = {'Heparin Flush (1000 units/mL)':'Heparin Flush'}, inplace = True)


# In[40]:

df['ethnicity']=df['ethnicity'].apply(race_reassign)        #bin ethnicities into subdemographics
df['religion']=df['religion'].apply(religion_reassign)      #bin religions into parent religion
df['time_of_stay']=df['time_of_stay'].dt.days               #convert time of stay into days
df['hadm_id']=df['hadm_id'].astype(int)                     #hadm_id to integer


# In[41]:

df


# In[42]:

x=0
list_of_meds=[]
while x<=8:
    list_of_meds.append(Meds.index[x])
    x=x+1


# In[43]:

list_of_meds


# In[46]:

df=df[['admission_location',
 'admission_type',
 'admittime',
 'diagnosis',
 'dischtime',
 'ethnicity',
 'hadm_id',
 'insurance',
 'marital_status',
 'religion',
 'subject_id','readmitted',
 'time_of_stay',
 'time_between_readmission',
 'number_of_readmits',
 'gender',
 'age','Penicillin G Benzathine',
 'Penicillin V Potassium',
 'Promethazine',
 'Orabase w/ Benzocaine Paste',
 'Aspirin Childrens',
 'Potassium Chl 40 mEq / 1000 mL D5NS',
 'Amitriptyline',
 'Medroxyprogesterone Acetate',
 'Oxycodone-Acetaminophen (5mg-325mg)']]


# medicines=['Lidocaine 1',
#  'Lansoprazole Oral Disintegrating Tab',
#  'DiphenhydrAMINE',
#  'Rifaximin',
#  'Albumin',
#  'Ondansetron ODT',
#  'Propranolol',
#  'Heparin Flush',
#  'Cholestyramine',
#  'Tacrolimus Suspension']
# meds={}
# for x in medicines:
#         
#         dataframe=pd.read_csv('/Drugs/Drugs/{}.csv'.format(x))
#         meds["{}".format(x)]=dataframe

# In[ ]:

for x in meds:
    meds[x]=meds[x].sort_values('subject_id')
    meds[x]=meds[x].reset_index(drop=True)
    meds[x]=meds[x][['subject_id','hadm_id']]
    meds[x]['{}'.format(x)]=np.ones(meds[x].shape[0])


# In[ ]:

for x in medicines:
    df=pd.merge(df,meds[x],on=['subject_id','hadm_id'],how='left')


# In[ ]:

df.loc[:,'Lidocaine 1':]=df.loc[:,'Lidocaine 1':].fillna(0)


# In[ ]:

df[df['subject_id']==507]


# In[48]:

df=df.drop_duplicates(keep='first')


# In[49]:

df['Number Of Drugs Taken']=df[list_of_meds].sum(axis=1)


# In[ ]:

df['Salmeterol'].value_counts()


# In[ ]:

df.loc[:,'Lidocaine 1':]=df.loc[:,'Lidocaine 1':].astype(int)


# In[50]:

df['AGE']=pd.cut(df['age'],[-1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110],labels=[0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22])


# In[51]:

df.drop(['age'],axis=1,inplace=True)


# In[ ]:

df=df[df['AGE'].notnull()]


# ## Data Visualizations

# In[ ]:

CHI2_vals= pd.DataFrame(index=list(df))
for x in list(df):
    CHI2_vals.loc[x,'chi2']=stats.chi2_contingency(pd.crosstab(df[x],df['readmitted']))[1].round(5)


# In[ ]:

CHI2_vals.sort_values('chi2')


# In[ ]:

decreasing_normalized_bar('admission_type')


# In[ ]:

decreasing_normalized_bar('admission_location')


# In[ ]:

decreasing_normalized_bar('religion')


# In[ ]:

decreasing_normalized_bar('insurance')


# In[ ]:

decreasing_normalized_bar('ethnicity')


# In[ ]:

decreasing_normalized_bar('gender')


# In[ ]:

decreasing_normalized_bar('Number Of Drugs Taken')


# In[ ]:

decreasing_normalized_bar('AGE')


# In[ ]:

decreasing_normalized_bar('marital_status')


# ### Evident that gender does not play a role. Hence, drop

# 
# ## Note: From external research and analysis, lab events _are_ *important* for the model but they are still dropped due to the given reason:
# ### Only 20 subject IDs are present in this dataset. For the purposes of this prototype, the training data derived from these would be to little to reach a conclusion. Therefore, they are dropped in this case but could be included in the final program

# ### drg_codes does not give us new information and is strongly correlated with the diagnosis. Hence,we get rid of it.

# In[52]:

d['procedures']


# In[53]:

procedures=pd.merge(d['procedures'],d['dIcdProcedures'],on='icd9_code',how='left')


# In[54]:

procedures


# In[ ]:

a=pd.merge(df,procedures,on='subject_id',how='left')


# In[ ]:

procedures


# In[55]:

procedures['icd9_code']=procedures['icd9_code'].astype(int)


# In[56]:

#procedure codes only require first 3 digits to classify, hence make a new column 'code'
for x in range(0,procedures.shape[0]):
    y=procedures.loc[x,'icd9_code']
    if y/1000>1:
        procedures.loc[x,'code']=y//100
    elif y/100>1:
        procedures.loc[x,'code']=y//10


# In[57]:

#To bunch up procedures as family of medicine they belong to
def procedure_classifier(x):
    if x==0:
        return 'INTERVENTIONS'
    elif 1<=x<=5:
        return 'OP ON NERVOUS SYSTEM'
    elif 6<=x<=7:
        return 'OP ON ENDOCRINE'
    elif 8<=x<=16:
        return 'OP ON EYE'
    elif x==17:
        return 'THERAPEUTIC'
    elif 18<=x<=20:
        return 'OP ON EAR'
    elif 21<=x<=29:
        return 'OP ON NOSE/MOUTH/PHARYHNX'
    elif 30<=x<=34:
        return 'RESPIRATORY'
    elif 35<=x<=39:
        return 'OP ON CARDIOVASCULAR'
    elif 40<=x<=41:
        return 'OP ON HEMIC/LYMPHATIC'
    elif 42<=x<=54:
        return 'OP ON DIGESTIVE'
    elif 55<=x<=59:
        return 'OP ON URINARY'
    elif 60<=x<=64:
        return 'OP ON MALE GENITALS'
    elif 65<=x<=71:
        return 'OP ON FEMALE GENITALS'
    elif 72<=x<=75:
        return 'OBSTETRIC'
    elif 76<=x<=84:
        return 'OP ON MUSCULOSKELETAL'
    elif 85<=x<=86:
        return 'OP ON INTEGUMENTARY'
    elif 87<=x<=99:
        return 'DIAGNOSTIC'


# In[58]:

procedures['code']=procedures['code'].apply(procedure_classifier)
proceduresconcat=procedures[['subject_id','hadm_id','code']]
proceduresfinal=pd.concat([proceduresconcat,pd.get_dummies(proceduresconcat['code'])],axis=1)
proceduresfinal.drop(['code'],axis=1,inplace=True)
proceduresfinal=proceduresfinal.drop_duplicates(keep='first').reset_index(drop=True)
proceduresfinal


# In[59]:

df=pd.merge(df,proceduresfinal,on=['subject_id','hadm_id'],how='left')


# In[ ]:

df


# #### admissions, patients, prescriptions,procedures_icd have been incorporated
# diagnoses_icd,drgcodes,labevents,icustays,inputevents,microb,services, transfers to be added

# In[60]:

diagnoses=pd.merge(d['diagnoses'],d['dIcdDiagnoses'],on='icd9_code',how='left')


# In[ ]:

df.loc[:,'DIAGNOSTIC':]=df.loc[:,'DIAGNOSTIC':].fillna(0)


# In[61]:

def restructure(x):
    try:
        #print(x)
        #y=diagnoses.loc[x,'icd9_code']
        z=int(x)
        if len(str(x))==5:
            return z//100
        elif len(str(x))==4:
            return z//10
    except:
        return x


# In[62]:

diagnoses['icd9_code']=diagnoses['icd9_code'].astype(str)


# In[63]:

d['dIcdDiagnoses'][d['dIcdDiagnoses']['long_title'].str.contains("chronic")]


# In[64]:

diagnoses.drop('short_title',axis=1,inplace=True)


# In[65]:

chronic=diagnoses[diagnoses['long_title'].notnull() & diagnoses['long_title'].str.contains("chronic")]

chronic['CHRONIC']=1


# In[66]:

chronic


# In[67]:

diagnoses['icd9_code']=diagnoses['icd9_code'].apply(restructure)


# In[68]:

def diagnoses_classifier(y):
    
    try:
        x = int(y)
        if 0<=x<=139:
            return 'Infec/Parasite'
        elif 140<=x<=239:
            return 'neoplasms'
        elif 240<=x<=279:
            return 'Endocrine/Nutriotional/Immunity'
        elif 280<=x<=289:
            return 'Blood Organs'
        elif 290<=x<=319:
            return 'Mental'
        elif 320<=x<=389:
            return 'Nervous System'
        elif 390<=x<=459:
            return 'Circulatory'
        elif 460<=x<=519:
            return 'Respiratory'
        elif 520<=x<=579:
            return 'Digestive'
        elif 580<=x<=629:
            return 'Genitourinary'
        elif 630<=x<=679:
            return 'Pregnancy'
        elif 680<=x<=709:
            return 'Skin'
        elif 710<=x<=739:
            return 'Connective Tissue'
        elif 740<=x<=759:
            return 'Congenital Anomalies'
        elif 760<=x<=779:
            return 'Perinatal'
        elif 780<=x<=799:
            return 'Ill-defined'
        elif 800<=x<=999:
            return 'Injury and Poisoning'
    except:
        return y


# In[69]:

for x in range(0,diagnoses.shape[0]):
    if 'V' in str(diagnoses.loc[x,'icd9_code']):
        diagnoses.loc[x,'icd9_code']='Hospital Contracted'
    elif 'E' in str(diagnoses.loc[x,'icd9_code']):
        diagnoses.loc[x,'icd9_code']='Accidents'


# In[70]:

diagnoses['icd9_code']=diagnoses['icd9_code'].apply(diagnoses_classifier)


# In[ ]:

diagnoses


# In[ ]:

a=pd.merge(a,diagnoses,on='subject_id',how='left')


# In[ ]:

a


# In[71]:

df=pd.merge(df,rows_to_columns(diagnoses.drop(['seq_num','long_title'],axis=1),'icd9_code'),on=['subject_id','hadm_id'],how='left')


# In[ ]:

df


# In[72]:

df=pd.merge(df,chronic.drop(['seq_num','icd9_code','long_title'],axis=1),on=['subject_id','hadm_id'],how='left')


# In[73]:

df.reset_index(drop=True)
df.loc[:,'Hospital Contracted':]=df.loc[:,'Hospital Contracted':].fillna(0)


# In[ ]:

df


# In[74]:

icu=d['icuStays'].drop(['dbsource','first_careunit','last_careunit','first_wardid','last_wardid','intime'],axis=1)


# In[75]:

for x in range(0,icu.shape[0]-1):
    try:
        if icu.loc[x,'hadm_id']==icu.loc[x+1,'hadm_id']:
            icu.loc[x,'totalicu']=icu.loc[x,'los']+icu.loc[x+1,'los']
            icu.loc[x+1,'totalicu']=icu.loc[x,'los']+icu.loc[x+1,'los']
            icu.drop(x+1,axis=0,inplace=True)
            icu=icu.reset_index(drop=True)
        else:
            icu.loc[x,'totalicu']=icu.loc[x,'los']
    except:
        pass


# In[76]:

df.columns = df.columns.str.replace('hadm_id_x','hadm_id')


# In[77]:

df=pd.merge(df,icu, how='left',on=['subject_id','hadm_id'])


# In[78]:

df['delta_t_icu_disch']=pd.to_datetime(df['dischtime'])-pd.to_datetime(df['outtime'])

df['delta_t_icu_disch']=df['delta_t_icu_disch'].dt.days

df=df[df['delta_t_icu_disch']>=0]

df.drop(['icustay_id'],axis=1,inplace=True)


# ### admissions, patients, prescriptions,d_icd_diagnoses,d_icd_procedures,d_lab_items,diagnoses_icd,drgcodes,labevents,procedures_icd, inputevents_cv, icustays have been incorporated
# 
# ### inputevents_mv, microbiologyevents, procedureevents_mv, services, transfers are yet to be added

# ### Our data is ready for being converted to our final model input. The new column has no need for patient identifiers.

# In[ ]:

list(df)


# In[79]:

df.to_csv('dataset.csv')


# In[80]:

#df.drop(['nan'],axis=1,inplace=True)
df=df.drop_duplicates(keep='first')

#Drop, as the unique ids do nothing to predict readmissions
df2=df.drop(['subject_id','admittime','dischtime','outtime','los','hadm_id'],axis=1)

#Reset readmitted to array of 0's
df2['readmitted']=0
df2=df2.reset_index(drop=True)

#Our target variable is now multiclass, 0 - No Readmissions/Extremely delayed, 1 - within 3 months, 2 - between 3 to 6 months

for x in range(0,df2.shape[0]):
    try:
        if 0<=df2.loc[x,'time_between_readmission']<=30:
            df2.loc[x,'readmitted']=1
        
        else:
            df2.loc[x,'readmitted']=0
    except:
        pass

#drop time as it is a troublesome variable to handle. Although, it conveys valuable information, there is no appropriate value it can be replaced by.
df2.drop('time_between_readmission',axis=1,inplace=True)

#OneHotEncode all categorical columns
df2=pd.concat([df2,pd.get_dummies(df2['marital_status'])],axis=1)
df2=pd.concat([df2,pd.get_dummies(df2['admission_type'])],axis=1)
df2=pd.concat([df2,pd.get_dummies(df2['admission_location'])],axis=1)
df2=pd.concat([df2,pd.get_dummies(df2['insurance'])],axis=1)
df2=pd.concat([df2,pd.get_dummies(df2['religion'])],axis=1)
df2=pd.concat([df2,pd.get_dummies(df2['ethnicity'])],axis=1)
df2=pd.concat([df2,pd.get_dummies(df2['gender'])],axis=1)

#Drop all the categ. columns
df2.drop(['marital_status','admission_type','admission_location','insurance','religion','ethnicity','diagnosis','gender'],axis=1,inplace=True)
df2=df2.drop('None',axis=1)

#Convert age to processable format
df2['AGE']=df2['AGE'].astype(int)
df2.drop('OTHER',axis=1,inplace=True)


# In[ ]:

list(df2)


# # Algorithms

# In[81]:

df.drop(['CHRONIC_x'],axis=1,inplace=True)


# In[ ]:

df.rename(columns={'CHRONIC_y': 'CHRONIC'}, inplace=True)


# In[88]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import cross_val_score

from sklearn.model_selection import train_test_split
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier  
from sklearn.metrics import roc_auc_score


# In[ ]:




# In[ ]:

np.set_printoptions(suppress=True,precision=2)


# #Define target variable
# Y=df2['readmitted']
# 
# #Split test train
# X_train, X_test, Y_train, Y_test = train_test_split(df2,Y,test_size=0.2)
# 
# #Remove target variable from feature data
# X_train.drop(['readmitted'],axis=1,inplace=True)
# X_test.drop(['readmitted'],axis=1,inplace=True)

# In[94]:

x=df2.drop(['readmitted'],axis=1)
y=df2['readmitted']


# In[93]:

df2=df2.fillna(0)


# In[85]:

import collections


# In[90]:

from sklearn.preprocessing import StandardScaler


# In[86]:

df2=df2.fillna(0)


# In[95]:

X_train, X_test, Y_train, Y_test = train_test_split(StandardScaler().fit_transform(x),y , test_size=0.2)


# In[ ]:

StandardScaler().fit_transform(x)


# In[ ]:

sklearn.version


# In[102]:

df2.to_csv('data.csv')


# In[1]:

import scipy
scipy.__version__


# In[2]:

from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE

smote = SMOTE(ratio='minority')
X_sm, Y_sm = smote.fit_sample(X_train,Y_train)
X_final, Y_final = smote.fit_sample(X_sm,Y_sm)

X_train=X_final

Y_train=Y_final


# In[ ]:

collections.Counter(Y_train)


# In[ ]:

X_train.shape


# In[ ]:

#Logistic Regression
LogReg = LogisticRegression(penalty='l1',class_weight={0:1,1:2})
LogReg.fit(X_train, Y_train)
Y_pred = LogReg.predict(X_test)
print('Accuracy:',LogReg.score(X_test,Y_test).round(2)*100,'%')


# In[ ]:

coefs=list(LogReg.coef_.ravel().round(2))


# In[ ]:

dict(list(zip(list(x),coefs)))


# In[ ]:

pickle.dump(LogReg, open('modelonemonthclass.sav', 'wb'))


# In[ ]:

pd.crosstab(Y_pred,Y_test,normalize='columns').round(3)


# In[ ]:

from sklearn.feature_selection import RFE


# In[ ]:

rfe = RFE(svc,5)


# In[ ]:

l=LogReg.coef_.ravel().round(2).tolist()


# In[ ]:

list(zip(l,x.columns))


# In[ ]:

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif["features"] = x.columns


# In[ ]:

vif


# In[ ]:

vif.sort_values('VIF Factor',ascending=False)


# In[ ]:

df2.corr()['ELECTIVE'].sort_values()


# In[ ]:

vif


# In[ ]:

df2.corr()['time_of_stay']


# In[ ]:

decreasing_normalized_bar('number_of_readmits')


# In[ ]:

a=list(zip(names,l))


# In[ ]:

a.sort(key=lambda x: x[1])


# In[ ]:

a


# In[ ]:

names=list(x.columns)


# In[ ]:

print(rfe.support_)
print(rfe.ranking_)


# In[ ]:




# In[ ]:

X_train


# In[ ]:

pd.crosstab(Y_pred,Y_test).round(2)


# In[ ]:

pd.crosstab(Y_pred,Y_test,normalize='columns').round(2)


# In[ ]:

from sklearn.metrics import classification_report
report = classification_report(Y_test, Y_pred)
print(report)


# In[ ]:

X_train


# In[ ]:

#Support Vector Machines
svc = SVC(probability=True,class_weight={0:1,1:150})
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_test,Y_test), 3)
acc_svc


# In[ ]:

svc.predict_proba(X_test[1,:].reshape(1,-1))


# In[ ]:

pd.crosstab(Y_pred,Y_test,normalize='columns').round(2)


# In[ ]:

y_=list(svc.predict_proba(X_test.head(1)).ravel().round(2))


# In[ ]:

sns.barplot(y=y_,x=[0,1,2])


# In[ ]:

l=list(LogReg.predict_proba(X_test[1,:].reshape(1,-1)).ravel())


# In[ ]:

LogReg.predict_proba(X_train.head(1))


# In[ ]:

l


# In[ ]:

sns.barplot(y=l,x=['No Readmits','One Month'])


# In[ ]:

plt.figure(figsize=(15,10))
plt.ylabel('Probability',fontsize=30)
plt.ylim(0,1)
plt.xticks(fontsize=14)
ax=sns.barplot(y=list(LogReg.predict_proba(X_test[1,:].reshape(1,-1)).ravel().round(2)),x=['No Readmission','Within One Month'])
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
    p.get_height()+0.02,
    '{:1.1f}%'.format(height*100),
    ha="center",color='black',fontsize=30) 



# In[ ]:

list(svc.predict_proba(X_test[1,:].reshape(1,-1)).ravel().round(2))


# In[ ]:

sns.barplot(y=list(svc.predict_proba(X_test[1,:]),x=['No Readmission','Readmitted within 3 months'])


# In[ ]:

plt.figure(figsize=(15,10))

plt.ylabel('Probability',fontsize=30)
plt.ylim(0,1)
plt.xticks(fontsize=14)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            p.get_height()+0.02,
            '{:1.1f}%'.format(height*100),
            ha="center",color='black',fontsize=30) 



# #Neural Networks 
# neural = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#        beta_2=0.999, early_stopping=False, epsilon=1e-08,
#        hidden_layer_sizes=(30, 30, 30), learning_rate='constant',
#        learning_rate_init=0.001, max_iter=200, momentum=0.9,
#        nesterovs_momentum=True, power_t=0.5, random_state=None,
#        shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
#        verbose=False, warm_start=False)
# neural.fit(X_train, Y_train)  
# Y_pred = neural.predict(X_test)  
# print(neural.score(X_test, Y_test).round(3))

# In[ ]:

neural.classes_


# In[ ]:

pd.crosstab(Y_pred,Y_test,normalize='columns').round(2)


# In[ ]:

neural.predict_proba(X_test[1,:].reshape(1,-1))


# #Random Forests
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42,class_weight={0:0.00000001,1:10000000,2:1000000})
# classifier.fit(X_train, Y_train)
# Y_pred=classifier.predict(X_test)
# classifier.score(X_test,Y_test).round(3)

# In[ ]:

from sklearn.metrics import classification_report
report = classification_report(Y_test, Y_pred)
print(report)


# In[ ]:

pd.crosstab(Y_pred,Y_test,normalize='columns').round(2)


# In[ ]:

classifier.classes_


# In[ ]:

list(classifier.predict_proba(X_test[22,:].reshape(1,-1)).ravel())


# In[ ]:

algorithms=[LogReg,neural,classifier,svc]


# In[ ]:

for x in algorithms:
    print(cross_val_score(x, df2.drop(['readmitted'],axis=1), df2['readmitted'], scoring='accuracy', cv = 10).mean() * 100)


# In[ ]:

d['admissions']


# In[ ]:

X_test.loc[[11]]


# In[ ]:

Y_test[11]


# In[ ]:

f=pd.concat([X_test,Y_test])


# In[ ]:

a=pd.concat([Y_test,X_test],axis=1)


# In[ ]:

a[a['readmitted']==1].reset_index(drop=True).loc[[1]]


# In[ ]:



