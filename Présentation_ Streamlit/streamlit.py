# import librairies
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px
import pickle 

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    classification_report,
)

from sklearn.model_selection import train_test_split, GridSearchCV
import streamlit as st 

df= pd.read_csv("/Users/kiki-wiwi/Downloads/archive (3)/train.csv")

st.title("Classification des personnes en tranches de crédit grace à l'IA")
st.subheader("Auteur: Ouiza MEBARKI")

st.sidebar.title("Sommaire")
pages= ["Contexte du projet", " Exploration de données", "Analyse de données", " Modélisation", "Conclusion"]

page= st.sidebar.radio("Aller vers la page: ", pages)

if page== pages[0]: 
    st.write("### Contexte du projet")

    st.write("Ce projet s'inscrit dans le contexte de crédit scoring, ce dernier consiste à analyser et évaluer le niveau de risque du demandeur de crédit en lui attribuant une note, autrement dit un score. ")
    #st.write ("**Probléme**: La direction de la société fianciére mondiale souhaite construire un systéme intelligent pour séparer les personnes en tranches de cédit.") 
    st.write ("**Objectif**: Nous avons à notre disposition le fichier credit score  qui contient des informations relative au crédit d'une personne à partir duquel nous devons créer un modèle d'apprentissage automatique capable de classer la cote de crédit. ")
    st.write (" **Aproche technique**: Premiérement on explore notre jeux de donées. Puis on visualise afin d'extraire des information et comprendre mieux notre dataset. Finalement on implémente des modéles de Machine Learniing pour prédire la cote crédit .")

    st.image("credit_score.jpg")


elif page== pages[1]: 

    st.write("### Exploreration de données")
    st.dataframe(df.head())

    st.write("Dimension du dataframe: ")
    st.write(df.shape)

    if st.checkbox("Afficher les valeus manquantes: "):
        st.write(df.isna().sum())

    if st.checkbox("Afficher les doublons: "):
        st.write(df.duplicated().sum())

    if st.checkbox("Discription des variables numériques: "):
        st.write(df.describe())

    if st.checkbox("Discription des variables catégoriques: "):
        st.write(df.describe(include='object'))
    

elif page== pages[2]:
    st.write("### Analyse de données")
    df_sample= pd.read_csv("df_sample.csv")

    st.subheader("*Objectif*: On regarde l'impact de chaque variable sur les tranches du crédit Score")

    st.subheader("Les proportions de la variablecible dans notre jeux de données ")

    cérdit_score_count= df_sample['Credit_Score'].value_counts()
    fig_sb, ax_sb= plt.subplots()
    ax_sb= plt.pie(cérdit_score_count, labels=cérdit_score_count.index, autopct='%1.2f%%')
    st.pyplot(fig_sb)
    

    st.subheader("Distribution de crédit score par rapport à l'occupation ")
    fig2= px.bar(df, x="Occupation", color="Credit_Score")
    st.plotly_chart(fig2)

    
    st.subheader("Distribution de crédit score par rapport à Payment_of_Min_Amount")
    fig_sb1, ax_sb1= plt.subplots()
    ax_sb1= sns.countplot(df_sample, x="Payment_of_Min_Amount", hue="Credit_Score")
    
    st.pyplot(fig_sb1)


    st.subheader("Distribution de crédit score par rapport à Crédit_mix")
    fig_sb2, ax_sb2= plt.subplots()
    ax_sb2=sns.countplot(df_sample, x="Credit_Mix", hue="Credit_Score")
    st.pyplot(fig_sb2)

    
    st.subheader("la corrélation entre les variables numeriques ")
    numericals = df_sample.select_dtypes(include='number').columns
    fig3, ax = plt.subplots(figsize=(20,15))
    sns.heatmap(df_sample[numericals].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1, ax = ax )
    st.write(fig3)

    st.subheader("la distribution des variables numeriques ")
    fig4, ax = plt.subplots( figsize=(8,20))
    df_sample.hist(bins=21,layout=(-1, 3), edgecolor="black", ax=ax)
    st.write(fig4)

    
elif page== pages[3]:
    st.write("### Modélisation ")
    df= pd.read_csv("df_preprocessed.csv")

    X= df.drop(['Credit_Score'],axis = 1)
    y=df['Credit_Score']
    
    # Split 
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=0)

    # scaler les données 
    scaler = StandardScaler()
    X_train= scaler.fit_transform(X_train)
    X_test= scaler.fit_transform(X_test)

    lr= pickle.load(open("model_lr.pkl", 'rb'))
    knn= pickle.load(open("model_knn.pkl", 'rb'))

    y_pred_lr= lr.predict(X_test)
    y_pred_knn= knn.predict(X_test)

    model= st.selectbox(label="Modèle", options= ['Logistic Regression', 'KNeighborsClassifier'])

    
    def train_model(model): 
        if model=='Logistic Regression': 
            y_pred= lr.predict(X_test)
           
        elif model== 'KNeighborsClassifier': 
            y_pred= knn.predict(X_test)

        f1= f1_score(y_test, y_pred, average='micro')

        target_names = ["class 0", "class 1", "class 2"]   
        st.dataframe(classification_report(y_test, y_pred, target_names=target_names, output_dict=True))
        #conf_mat= confusion_matrix(y_test,lr.predict(X_test) )
        #fig_m,ax= plt.subplot
        #sns.heatmap(conf_mat, annot=True, cmap='Blues', ax=ax)
        #st.write(fig_m)

        return  f1
    
    st.write("La précison du modèle est: ", train_model(model))
    

    
    







    
   
    



    





