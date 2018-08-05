from django.shortcuts import render
from django.http import HttpResponse
from django import forms
import pandas as pd
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from matplotlib.gridspec import GridSpec
import matplotlib
sns.set()

pd.options.display.max_columns = None


admission_locations = ['PHYS REFERRAL/NORMAL DELI',
 'TRANSFER FROM HOSP/EXTRAM',
 'CLINIC REFERRAL/PREMATURE',
 'EMERGENCY ROOM ADMIT',
 'TRANSFER FROM SKILLED NUR',
 'TRANSFER FROM OTHER HEALT']
procedures = ['Aortocor bypas-3 cor art',
 '1 int mam-cor art bypass',
 'Extracorporeal circulat',
 'Left heart cardiac cath',
 'Lt heart angiocardiogram',
 'Aortocor bypas-2 cor art',
 'Rt/left heart card cath',
 'Coronar arteriogr-2 cath',
 'Venous cath NEC',
 'Packed cell transfusion']
admission_type = ['ELECTIVE', 'EMERGENCY', 'URGENT']
ethnicity = [{"name":'White',"val":'WHITE'}, {"name":'Others', "val":'OTHERS'}, {"name":'Hispanic',"val":"HISPANIC"},{"name":'Black' ,"val":'BLACK'}]
insurance = ['Medicare', 'Private', 'Self Pay', 'Government', 'Medicaid']
marital_status = ['MARRIED', 'SINGLE', 'DIVORCED', 'WIDOWED', 'SEPARATED', 'UNKNOWN (DEFAULT)']
religion = ['CATHOLIC', 'PROTESTANT QUAKER', 'NOT SPECIFIED', 'OTHER', 'JEWISH']
gender = ['Male', 'Female']

model_path = ""
total_cost_predictor_model_name = "Total_cost_predictor.pkl"
gap_predictor_model_name = "Gap_predictor.pkl"
Dataset_path = ""
total_cost_predictor = joblib.load(model_path+total_cost_predictor_model_name)
gap_predictor = joblib.load(model_path+gap_predictor_model_name)

response_dict = {'admission_locations' : admission_locations , "ethnicity":ethnicity ,'procedures':procedures}
# Create your views here.
def index(request):
    return render(request,"index.html")

def forms(request):
    return render(request,"form.html" ,response_dict)


def process(request):
    columns = ['ICU_LOS', '1 int mam-cor art bypass', 'Aortocor bypas-2 cor art',
       'Aortocor bypas-3 cor art', 'Coronar arteriogr-2 cath',
       'Extracorporeal circulat', 'Left heart cardiac cath',
       'Lt heart angiocardiogram', 'Packed cell transfusion',
       'Rt/left heart card cath', 'Venous cath NEC',
       'CLINIC REFERRAL/PREMATURE', 'EMERGENCY ROOM ADMIT',
       'PHYS REFERRAL/NORMAL DELI', 'TRANSFER FROM HOSP/EXTRAM',
       'TRANSFER FROM OTHER HEALT', 'TRANSFER FROM SKILLED NUR', 'ELECTIVE',
       'EMERGENCY', 'URGENT', 'BLACK', 'HISPANIC', 'OTHERS', 'WHITE',
       'Government', 'Medicaid', 'Medicare', 'Private', 'Self Pay', 'DIVORCED',
       'MARRIED', 'SEPARATED', 'SINGLE', 'UNKNOWN (DEFAULT)', 'WIDOWED',
       'CATHOLIC', 'JEWISH', 'NOT SPECIFIED', 'OTHER', 'PROTESTANT QUAKER',
       'Female', 'Male']
    age_bins = [ 23.923,  31.7  ,  39.4  ,  47.1  ,  54.8  ,  62.5  ,  70.2  , 77.9  ,  85.6  ,  93.3 ]
    length_bins = [-0.035,  3.5  ,  7.   , 10.5  , 14.   , 17.5  , 21.   , 24.5  ,  28.   , 31.5  , 35.   ]

    new_dataframe = pd.DataFrame(columns=columns)
    print(request.GET)
    for y in request.GET.values():
        if y in new_dataframe.columns:
            new_dataframe.loc[0,y] = 1
    age ,los =0,0
    try:
        age = int(request.GET['age'])
        los = int(request.GET['los'])
    except:
        print(age,los)
    new_dataframe = pd.concat((new_dataframe,pd.get_dummies(pd.cut([age],bins=age_bins))),axis=1)
    new_dataframe = pd.concat((new_dataframe,pd.get_dummies(pd.cut([los],bins=length_bins))),axis=1)
    new_dataframe.fillna(0,inplace=True)
    #print(new_dataframe)
    total_cost = total_cost_predictor.predict(new_dataframe)[0]
    gap = gap_predictor.predict(new_dataframe)[0]
    print(insurance)
    if request.GET['insurance']=="Self Pay":
        print("GAP",0)
        gap = 0
    total_cost,gap=max(total_cost,gap),min(total_cost,gap)
    payable = total_cost - gap

    b=pd.DataFrame({'Source':['Total Cost',"Paid by insurance","Paid by claimant"],"Amount":[total_cost,gap,total_cost-gap]})
    plt.figure(2, figsize=(14,10))
    the_grid = GridSpec(2, 2)

    plt.subplot(the_grid[0, 0],  title='')
    sns.barplot(x='Source',y='Amount',data=b,palette='Spectral')
    plt.xlabel('')
    plt.subplot(the_grid[0, 1], title='')

    plt.pie(b["Amount"].iloc[1:],labels=b["Source"].iloc[1:],autopct='%1.1f%%')

    plt.suptitle('Cost distribution', fontsize=16)
    '''plt.show()

    plt.pie([gap,payable],labels=["Total cost","Payable"],autopct='%1.1f%%')'''

    buf = BytesIO()
    plt.savefig(buf, format='png')
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
    buf.close()
    response_dict['contents']=image_base64
    response_dict['total_cost'] = int(round(total_cost))
    response_dict['payable'] = int(round(payable))
    response_dict['gap'] = int(round(gap))
    return render(request,"results.html",response_dict)

def results(request):
    return render(request,"results.html",response_dict)
