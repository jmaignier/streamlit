#!/usr/bin/env python
# coding: utf-8

# In[1]:

## Packages
import streamlit as st
import streamlit.components.v1 as components
from streamlit.components.v1 import html
import random
import pandas as pd
import numpy as np
import unicodedata

def decode(s):
    return unicodedata.normalize('NFD', s).encode('ascii', 'ignore').lower()

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def load_quizz(drop):
    quizz =  pd.read_csv('https://raw.githubusercontent.com/maigje98/test_app/master/Quizz.csv',sep=';',engine='python')
    quizz['DIFFICULTE'] = quizz['DIFFICULTE'].astype(str)
    quizz = quizz.drop(32)
    return quizz.drop(drop)


  
def main():
    dict_diff = {None:None,'1':'Facile','2':'Moyen','3':'Dure'}
    dict_pts = {'1':'1 point','1.5':'2 points','2':'3 points','2.5':'4 points','3':'5 points'}
    
    st.title("Bienvenue dans le Super Quizz de Culture G (rémy) 🧠")
    rd = st.slider("Choisis un nombre 'au hasard'",min_value=0,max_value=100,value=0,step=1)
    random.seed(rd)
    quizz = load_quizz([])
    pts = [int(o[0]) for o in range(quizz.shape[0]) for o in list(dict_pts.values())]
    questions_repondues = st.sidebar.multiselect("Questions auxquelles vous avez déjà répondu",quizz.index)
    total_pts = st.sidebar.multiselect("Nombre de points",pts)
    st.sidebar.markdown(f"""<div> <span style='font-family:Arial Black;font-size=15px;color:green'> Ton score est pour l'instant de :
    <br> {np.sum(total_pts)} points""",unsafe_allow_html=True)
    
    quizz = load_quizz(questions_repondues)
    st.sidebar.info(f"{quizz.shape[0]} questions restantes")

    diff = st.sidebar.selectbox(f"Question : Choisis ta difficulté",quizz.DIFFICULTE.unique(),format_func=lambda x:dict_diff[x])
    qu = random.choice(quizz.query(f'DIFFICULTE=={[diff]}').index)
    vrai = quizz.loc[qu]['VRAI']
    
    st.markdown(f"""<div> <span style='font-family:Arial Black;font-size=18px;color:steelblue'> <br> Numéro {qu} <br>
    Question {dict_diff[diff]}   <br> {quizz.loc[qu][0]} <br> </span></div> <br>""",unsafe_allow_html=True)

    all_reps = quizz.loc[qu][1:5].dropna().tolist()
    random.shuffle(all_reps)

    if len(all_reps)>1 and diff in ['2','3']:
        choice = st.selectbox("Duo / Carré / Cash",["Duo","Carré","Cash"],index=2)
        col,row = st.beta_columns([1,0.001])
        with  col.beta_expander("Faire votre réponse"):
            if choice == 'Cash':
                rep = st.text_input("Cash : Entrez directement la réponse")
                diff = diff
            elif choice == 'Duo':
                diff = str(int(diff)-1)
                reps = [vrai]+ [random.choice([rep for rep in all_reps if rep!=vrai])]
                random.shuffle(reps)
                rep = st.radio("Choisis la bonne réponse",reps)
            elif choice == 'Carré':
                diff = str(int(diff)-0.5)
                reps = all_reps
                rep = st.radio("Choisis la bonne réponse",reps)
                
    elif diff == '1' and len(all_reps)>1 :
        st.info("Carré seulement disponible")
        col,row = st.beta_columns([1,0.001])
        with col.beta_expander("Faire votre réponse"):
            rep = st.radio("Choisis la bonne réponse",all_reps)
    else:
        col,row = st.beta_columns([1,0.001])
        with col.beta_expander("Faire votre réponse"):
            rep = st.text_input("Cash seulement disponible : Entrez directement la réponse")
            
    if st.button(f"Valider réponse"):
        if rep == vrai or rep in vrai or decode(rep) in decode(vrai):
            col.success(f"Félications ! 🎉  {rep}, Bonne réponse  !  +{dict_pts[diff]}, ajoute ton score aux précédents")
        else:
            col.error(f"Aïe .. {rep}, Mauvaise réponse ...")
            col.warning(f"La réponse était {vrai}")
                
    html("""Fait par <a href="https://github.com/maigje98/test_app/" target="_blank"> Jeremy Maignier </a> (cliquer pour ouvrir le lien dans un nouvel onglet)""")
    
    
if __name__ == "__main__":
    main()
