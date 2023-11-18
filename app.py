import streamlit as st
import numpy as np 
import pickle



st.title("Prédiction de score de crédit")
st.subheader("Application réalisée par Ouiza MEBARKI")

# Chargement du model 
model= pickle.load(open("model.pkl", 'rb'))
# Définition d'une fonction d'inférence
def inference(Annual_Income, Outstanding_Debt, Credit_Mix,Payment_of_Min_Amount, Interest_Rate, Num_of_Loan):
    new_data= np.array([Annual_Income, Outstanding_Debt, Credit_Mix,Payment_of_Min_Amount, Interest_Rate, Num_of_Loan,Num_Credit_Card, Credit_History_Age])
      
    pred= model.predict(new_data.reshape(1,-1))
    return pred
# L'utilisation saisie une valeur pour chaque caractéristique 

Annual_Income = st.number_input(label= 'Annual_Income', value= 50000)
Outstanding_Debt= st.number_input(label= 'Outstanding_Debt', value= 200)
Credit_Mix= st.number_input(label= 'Credit_Mix', min_value= 0, max_value= 2, value= 1)
Payment_of_Min_Amount= st.number_input(label= 'Payment_of_Min_Amount', min_value= 0, max_value= 2, value= 2)
Interest_Rate = st.number_input(label= 'Interest_Rate', min_value= 0, max_value= 34, value= 10)
Num_of_Loan= st.number_input(label= 'Num_of_Loan', min_value= 0, max_value= 9, value= 2)
Num_Credit_Card= st.number_input(label= 'Num_Credit_Card', min_value= 0, max_value= 11, value= 0)
Credit_History_Age= st.number_input(label= 'Credit_History_Age', value= 100)

# Création du bouton "Prédict" qui retourne la prédiction du modéle 

if st.button("Predict"):
    prediction= inference(Annual_Income, Outstanding_Debt, Credit_Mix,Payment_of_Min_Amount, Interest_Rate, Num_of_Loan)
    
    if prediction[0]==0:
        resultat= " Le score de credit est : mauvais"

    elif prediction[0]==1:
         resultat= " Le score de credit est : Bon"

    else: 
        resultat= " Le score de credit est : Trés bon"


    
    #resultat= " Le score de credit est: ",prediction[0] 
    
    st.success(resultat)

    