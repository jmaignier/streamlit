#!/usr/bin/env python
# coding: utf-8

# In[1]:

## Pandas packages
import streamlit as st
import random
import pandas as pd
import numpy as np
## Visu

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def load_quizz(drop):
    quizz =  pd.read_csv('/Users/jeremymaignier/Desktop/Loisirs/Quizz.csv',sep=';')
    quizz['DIFFICULTE'] = quizz['DIFFICULTE'].astype(str)
    
    return quizz.drop(drop)


  
def main():
    dict_diff = {None:None,'1':'Facile','2':'Moyen','3':'Dure'}
    dict_pts = {'1':'1 point','1.5':'2 points','2':'3 points','2.5':'4 points','3':'5 points'}
    
    st.title("Bienvenue dans le Super Quizz de Culture G (rémy) 🧠")
    score = 0
    random.seed(30)
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
        
        with st.beta_expander("Faire votre réponse"):
            rep = st.radio("Choisis la bonne réponse",all_reps)
    else:
        
        with st.beta_expander("Faire votre réponse"):
            rep = st.text_input("Cash seulement disponible : Entrez directement la réponse")
            
    if st.button(f"Valider réponse"):
        if rep == vrai or rep in vrai:
            st.success(f"Félications ! 🎉  {rep}, Bonne réponse  !  +{dict_pts[diff]}, ajoute ton score aux précédents")
        else:
            st.error(f"Aïe .. {rep}, Mauvaise réponse ..")
            st.warning(f"La réponse était {vrai}")
                
    
if __name__ == "__main__":
    main()
