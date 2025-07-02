# -*- coding: utf-8 -*-
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np


#loading the models
diabetes_scaler = pickle.load(open("diabetes_scaler.sav",'rb'))
diabetes_model = pickle.load(open("diabetes_model.sav",'rb'))

parkinson_scaler = pickle.load(open("parkinsons_scaler.sav",'rb'))
parkinson_model = pickle.load(open("parkinsons_model.sav",'rb'))

heart_disease_model = pickle.load(open("heartdisease_model.sav",'rb'))


def diabetes_pred(input_data):
    #changing the imnput data to a nummpy array
    input_array = np.asarray(input_data)

    #reshape the array as we are predicting for one instance
    reshaped_input = input_array.reshape(1,-1)

    #data standardization
    std_data = diabetes_scaler.transform(reshaped_input)
    print(std_data)

    prediction = diabetes_model.predict(std_data)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def parkinsons_pred(input_data):
    #Changing input data into numpy array
    input_array = np.asarray(input_data)
    
    #reshape the numpu array
    input_reshaped = input_array.reshape(1,-1)
    
    #standardizing the data
    std_data = parkinson_scaler.transform(input_reshaped)
    
    prediction = parkinson_model.predict(std_data)
    print(prediction)
    
    
    if (prediction[0] == 0):
      return "The person doesn't have Parkinson's Disease" 
    
    else:
      return "The person have Parkinson's Disease"

with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                            ['Diabetes Prediction',
                             'Heart Disease Prediction',
                             'Parkinson\'s Prediction'],
                            icons=['activity', 'heart', 'person'],
                            default_index = 0)
    
#Diabetes prediction page
if (selected == 'Diabetes Prediction'):
    
    #page title
    st.title('Diabetes Prediction Using ML')
    #getting the input data from the user   
    #columns for input stream
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    
    with col2:
        Glucose = st.text_input('Glucose Level')
        
    with col3:
        BloodPressure = st.text_input('Blood Pressure Value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness Value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI Value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    
    with col2:
        Age= st.text_input('Age of the Person')
        
     
    #code for Prediction
    diab_diagnosis = ''
    
    #Creating a button for prediction
    if st.button('Diabetes Test Result'):
        try:
           Pregnancies = int(Pregnancies)
           Glucose = int(Glucose)
           BloodPressure = int(BloodPressure)
           SkinThickness = int(SkinThickness)
           Insulin = int(Insulin)
           BMI = float(BMI)
           DiabetesPedigreeFunction = float(DiabetesPedigreeFunction)
           Age = int(Age)
           
           diab_diagnosis =  diabetes_pred([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
           st.success(diab_diagnosis)
        
        except ValueError:
            st.error("Please enter valid **numeric values** in all input fields.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

#Heart Disease prediction page
if (selected == 'Heart Disease Prediction'):
    
    #page title
    st.title('Heart Disease Prediction Using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest pain types')
        
    with col1:
        trestbps = st.text_input('Resting blood pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting blood sugar > 120mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
   
    heart_diagnosis = ''
    
    #Creating a button for prediction
    if st.button('Heart Disease Test Result'):
        try:
            age = int(age)
            sex = int(sex)
            cp = int(cp)
            trestbps = float(trestbps)
            chol = float(chol)
            fbs = int(fbs)
            restecg = int(restecg)
            thalach = float(thalach)
            exang = int(exang)
            oldpeak = float(oldpeak)
            slope = int(slope)
            ca = int(ca)
            thal = int(thal)

            input_data = [[age, sex, cp, trestbps, chol, fbs, restecg,
                       thalach, exang, oldpeak, slope, ca, thal]]

            heart_prediction = heart_disease_model.predict(input_data)
            if (heart_prediction[0] == 1):
                heart_diagnosis = 'The person is having heart disease'
            else:
                heart_diagnosis = 'The person does not have any heart disease'
            st.success(heart_diagnosis)


        except ValueError:
            st.error("Please enter valid **numeric values** in all input fields.")
            
#Parkinson's prediction page
if (selected == 'Parkinson\'s Prediction'):
    
    #page title
    st.title('Prkinson\'s Prediction Using ML')
    
    col1,col2,col3,col4,col5 = st.columns(5)
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')

    
    #code for prediction
    parkinsons_diagnosis = ''

        
    #creating a button for prediction
    
    if st.button("Parkinson's Test Result"):
        try:
            
            
            input = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]
            input_data = [float(i) for i in input]
            parkinsons_diagnosis = parkinsons_pred(input_data)

            st.success(parkinsons_diagnosis)
        except ValueError:
            st.error("Please enter valid **numeric values** in all input fields.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
    



















