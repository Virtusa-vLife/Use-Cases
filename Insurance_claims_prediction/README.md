The purpose of this project is to design a model that will predict the insurance amount that a patient can claim for the treatment of a particular disease. Using the mimic database which contains the electronic health records of patients and the claims database which contains the insurance claim records, we will generate a model that will predict the total cost incurred and the amount that can be claimed. This will help the patients get a better picture of the expenditure for the treatment of a disease.

For this purpose, first, an analysis is performed over the claims dataset to get a clear picture of the features involved.
Next, the most important features that are common with the electronic health records is determind using stastical test.
Once the most important features are found out, we used these features to merge the electronic health records with the claims records to impute the total cost of the disease into the electronic health records
After the merging of the databases, we predict the total cost and the payable amount using the electronic health records of the patients.

Files:-

1)Analysis_of_claims_dataset is the python notebook used to perform the analysis of the insurance claims dataset

2)Final_dataset_generation is the python notebook used to impute the total cost and payable amount into the electronicc health records.

3)Statistical_analysis_ehr_claims_final_dataset is the notebook used to perform statistical test and build machine learning models to predict the total cost of the treatment of the disease and the payable amount.

Python module requirements:-

1)sklearn

2)pandas

3)numpy

4)matplotlib

5)seaborn

6)Django

For the statistical analysis the following additional modules are required:-

7)scipy

8)statsmodels

Steps to generate the dataset and run the user interface:-

1)Analysis_of_claims_dataset will give an analysis of the insurance claims database and generate a csv file which will be used to merge with the electronic health records.

2)Final_dataset_generation can be used to merge the insurance claims and the electronic records. This will generate the final dataset with the insurance costs imputed into the electronic health records.

3)Statistical_analysis_ehr_claims_final_dataset is the python notebook used to analyse the final dataset and build the predictive models.

4)The Django project used to create the server is located in the Gui folder. In order to run the server, browse to Gui folder(where manage.py file is located) and run the command "python manage.py runserver"

5)Once the server has started, go to browser and access localhost port number 8000(127.0.0.1:8000). 
