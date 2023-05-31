import pickle
import pandas as pd 
import numpy as np


dictionary={'DM':1, 'LD':2, 'DSA':3, 'OOP':4, 'BCN':5, 'M3':6, 'PA':7, 'DMSL':8, 'CG':9, 'SE':10, 'M1':11, 'PH':12, 'SME':13,'BE':14, 'PPS':15, 'CH':16, 'M2':17, 'SM':18, 'MECH':19, 'PHY':20}
dictionary2={'Distinction':1,'First':2,'Fail':3}
print(type(dictionary))
#Using the latest weights to predict the results
with open('final_model.pkl', 'rb') as file:
    model = pickle.load(file)



PRNNO=input("Enter PRN NO.:")
SUBJECT=input("Enter Subject:")
INSEM=int(input("Enter Insem Marks:"))
TW=int(input("Enter TW marks:"))
PR=int(input("Enter Practical marks:"))
subjectcode=dictionary[SUBJECT]

output=model.predict([[INSEM,TW,PR,subjectcode]])
confidance=round(100*(np.max(output[0])),2)
output2=int(output)
if output==1:
	print('GRADE:Distinction');
elif output==2:
	print('GRADE:First Class')
elif output==2:
	print('GRADE:Fail')
print("CONFIDENCE:",confidance)

