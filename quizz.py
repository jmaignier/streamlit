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


@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def load_quizz(drop):
    quizz =  pd.read_csv('https://raw.githubusercontent.com/maigje98/test_app/master/Quizz.csv?token=APV2NUAN525NZLPDFJKXJVS7T4GV2',sep=';')
    quizz['DIFFICULTE'] = quizz['DIFFICULTE'].astype(str)
    quizz=quizz.drop(32)
    
    return quizz.drop(drop)

def decode(s):
    return unicodedata.normalize('NFD', s).encode('ascii', 'ignore').lower()

  
def main():
    
    st.markdown("""
    <head>
        <title> Style CSS </title>
        <style type="text/css">
        body {background-color:;
              font-family:Apple Chancery;
              font-size:18px;
              }
        </style>
    </head>""",unsafe_allow_html=True)

     
     
    dict_diff = {None:None,'1':'Facile','2':'Moyen','3':'Dure'}
    dict_pts = {'1':'1 point','1.5':'2 points','2':'3 points','2.5':'4 points','3':'5 points'}
    
    html(" <div> <h1 style='font-family:Snell Roundhand'> Bienvenue dans le Super Quizz de Culture G (rémy) 🧠")
    rd = st.slider("Choisis 'au hasard' un nombre au début et ne le modifie plus",min_value=0,max_value=100,step=1,value=0)
    random.seed(rd)
    st.markdown('---')
    quizz = load_quizz([])
    pts = [int(o[0]) for o in range(quizz.shape[0]) for o in list(dict_pts.values())]
    questions_repondues = st.sidebar.multiselect("Questions auxquelles vous avez déjà répondu",quizz.index)
    
    total_pts = st.sidebar.multiselect("Nombre de points",pts)
    st.sidebar.markdown(f"""<div> <span style='font-family:Arial Black;font-size:17px;color:olive'> Ton score est pour l'instant de :
    <br> {int(np.sum(total_pts))} points </span> <br /> </div>""",unsafe_allow_html=True)
    
    quizz = load_quizz(questions_repondues)
    st.sidebar.markdown('---')
    
    st.sidebar.markdown(f"""<div style='background-color:Aliceblue;font-family:cursive;
    font-size:20px;
    padding-top: 10px;
    padding-right: 10px;
    padding-bottom: 10px;
    padding-left: 10px;'> {quizz.shape[0]} questions restantes <br> </div>""",unsafe_allow_html=True)
    st.sidebar.markdown(" <br> ",unsafe_allow_html=True)

    diff = st.sidebar.selectbox(f"Question : Choisis ta difficulté",quizz.DIFFICULTE.unique(),format_func=lambda x:dict_diff[x])
    qu = random.choice(quizz.query(f'DIFFICULTE=={[diff]}').index)
    vrai = quizz.loc[qu]['VRAI']
    
    with st.beta_expander("Afficher la question"):
        st.markdown(f"""<div> <span style='font-family:Luminari;font-size:22px;color:steelblue'> <u>Numéro {qu}</u> :
        Question {dict_diff[diff]}   <br> {quizz.loc[qu][0]} <br> </span></div> <br>""",unsafe_allow_html=True)

        all_reps = quizz.loc[qu][1:5].dropna().tolist()
        random.shuffle(all_reps)

        if len(all_reps)>1 and diff in ['2','3']:
            options = ["Cash","Duo","Carré"]
        elif diff == '1' and len(all_reps)>1 :
            options = ["Carré"]
        else :
            options = ["Cash"]
        
    st.markdown("""<style> div.Widget.row-widget.stSelectbox div{flex-direction:row;font-family: Arial Black ;color:steelblue;align-items: flex-start;
     }
    </style>""", unsafe_allow_html=True)

    st.markdown("""<style> div.Widget.row-widget.stRadio div{flex-direction:row;font-family: Arial Black ;color:green;align-items: flex-start;
     }
    </style>""", unsafe_allow_html=True)

    choice = st.selectbox("Duo / Carré / Cash",options)
    st.markdown('---')
    with  st.beta_expander("Faire votre réponse"):
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
                
            
    if st.button(f"Valider réponse"):
        if rep == vrai or rep in vrai or decode(rep) in decode(vrai):
            st.success(f"Félications ! 🎉  {vrai}, Bonne réponse  !  +{dict_pts[diff]}, ajoute ton score aux précédents")
        else:
            st.error(f"Aïe .. {rep}, Mauvaise réponse ... 😟 \n La réponse était {vrai}")
            
                
        st.markdown("""<div style='background-color:Salmon;font-family:Calibri;
        padding-top: 10px;
        padding-right: 10px;
        padding-bottom: 10px;
        padding-left: 10px;'> Merci de cliquer sur <b> - (à droite de 'faire votre réponse') </b> pour <b>cacher</b> la prochaine question <br> Enregistre le numéro de cette question et tes points obtenus en haut à gauche pour passer à la suivante </div>""",unsafe_allow_html=True)
        
    st.markdown('---')
    html("""<br> <br> Réalisé par <a href="https://github.com/maigje98/test_app/" target="_blank"> Jérémy Maignier </a> (cliquer pour ouvrir le lien dans un nouvel onglet)""")
        
        
    
if __name__ == "__main__":
    main()
