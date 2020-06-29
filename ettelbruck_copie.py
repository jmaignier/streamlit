#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
from streamlit import caching
import urllib.request as urllib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
import datetime as dt
from datetime import datetime,timedelta, date
import seaborn as sns
import ast
import time
import os
from os import listdir
from os.path import isfile, join
import altair as alt
import matplotlib.dates as mdates
from matplotlib import ticker
from bokeh.plotting import figure
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import jours_feries_france as jff
from jours_feries_france import JoursFeries
import pydeck as pdk
import plotly_express as px
import plotly.graph_objects as go
import random
from plotly import tools
from plotly.subplots import make_subplots


############################ FONCTIONS UTILES ########################

def change_type(o):
    try :
        return int(o)
    except :
        return o
        
def conv_period(x):
    if 7<=x<11:
        return "Matin"
    elif 11<=x<15:
        return "Midi"
    elif 17<=x<22 :
        return "Soir"
    else:
        return "autre"
        
############################## DONNEES URL ###########################

def load_data_brute():
    url='http://admin.parkoview.com/input/sta.csv'
    data_brute=pd.read_csv(url,sep=';',header=None,error_bad_lines=False)
    data_brute.columns=['date']+[o for o in range(1,16)]+[o for o in range(101,128)]
    data_brute.dropna(inplace=True)
    data_brute['date']=pd.to_datetime(data_brute['date'],format='%Y-%m-%d %H:%M:%S',errors='coerce')
    data_brute.dropna(inplace=True)
    #data_brute['date']=data_brute['date'].apply(lambda x:datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
    data_brute.set_axis([change_type(o) for o in data_brute.columns], axis=1, inplace=True)
    
    places=[o for o in data_brute.columns if type(o)==int]
    nbre_places=len(places)
    nbre_dates=data_brute.shape[0]
    #for o in places:
        #pkg[o]=pkg[o].astype(float)
    for o in places:
        data_brute[o]=data_brute[o].astype(int)
        data_brute[o]=data_brute[o].astype(str)

    data_brute['taux_occup']=[round((np.count_nonzero(data_brute[places].iloc[o]!=str(0))/nbre_places)*100,2) for o in range(nbre_dates)]
            
    return data_brute
 ########################################
 
def process_pkg(data_brute):
    temp=data_brute.copy()
    temp['date']=pd.to_datetime(temp['date'])
    temp.set_index('date',inplace=True)
    #temp['prec']=temp['date'].apply(lambda x :datetime.strftime(x,"%Y-%m-%d %H:%M"))
    #pkg=temp.groupby(['prec'],sort=True).agg({"taux_occup":'max'})
    #pkg=pkg.reset_index()
    #pkg=pkg.rename(columns={'prec':'date'})
    #pkg['date']=pd.to_datetime(pkg['date'])
    pkg=temp[['taux_occup']].resample('60S').mean().fillna(method='ffill')
    #pkg['taux_occup']=pkg['taux_occup'].fillna(method='ffill')
    pkg.reset_index(inplace=True)
    #for o in places:
        #pkg[o]=pkg[o].astype(float)
    pkg['jour'] = pkg['date'].dt.day_name()
    pkg['jour_num']=pkg['date'].dt.day
    pkg['mois']=pkg['date'].dt.month_name()
    pkg['mois_num']=pkg['date'].dt.month
    pkg['semaine']=pkg['date'].dt.week
    pkg['jour_semaine']=pkg['date'].apply(lambda x: x.isoweekday())
    pkg['periode']=pkg['date'].dt.hour.apply(conv_period)
    
    return pkg
    
####################################################


def process_sta(data_brute):
    data_utile=data_brute.copy()
    places=[o for o in data_utile.columns if type(o)==int]
    
    data_utile.drop('taux_occup',axis=1,inplace=True)
    data_utile['date']=pd.to_datetime(data_utile['date'])
    
    data_utile['jour']=data_utile['date'].dt.date
    
    liste_jour=data_utile['jour'].unique()
    data_utile.set_index('date',inplace=True)
    for o in places:
        data_utile[o]=data_utile[o].astype(float)
    
    grand=[]
    for j in liste_jour:
        test=data_utile[data_utile['jour']==j][places]
        #shift=test.shift(1)
        #shift.drop('jour',axis=1,inplace=True)
        #test.drop('jour',axis=1,inplace=True)
        petit=[]
        for o in test.columns:
            temp=test[o].diff()
            temp_bis=temp[temp!=0]
            if temp_bis.shape[0]>2:
                if temp_bis.shape[0]%2==0:
                    if temp_bis.iloc[1]>0.0:
                        moy=np.mean([temp_bis.index[2*(o+1)]-temp_bis.index[2*o+1] for o in range(temp_bis.shape[0]//2-1)])
                    if temp_bis.iloc[1]<0.0:
                        moy=np.mean([temp_bis.index[2*o+1]-temp_bis.index[2*o] for o in range(temp_bis.shape[0]//2)])
                elif temp_bis.shape[0]%2!=0.0:
                    if temp_bis.iloc[1]>0.0:
                        moy=np.mean([temp_bis.index[2*(o+1)]-temp_bis.index[2*o+1] for o in range(temp_bis.shape[0]//2)])
                    if temp_bis.iloc[1]<0.0:
                        moy=np.mean([temp_bis.index[2*o+1]-temp_bis.index[2*o] for o in range(temp_bis.shape[0]//2)])
            elif temp_bis.shape[0]==2:
                if temp_bis.iloc[1]>0.0:
                    moy=test.index[-1]-temp_bis.index[1]
                if temp_bis.iloc[1]<0.0:
                    moy=temp_bis.index[1]-temp_bis.index[0]
            else:
                if test[o].iloc[0]!=0.0:
                    moy=test.index[-1]-test.index[0]
                else:
                    moy=timedelta(hours=0,minutes=0,seconds=0)
            petit.append(moy)
                
        grand.append(petit)
    station=pd.DataFrame(grand)
        
    station.set_axis(places,axis=1,inplace=True)
    station.set_index(pd.to_datetime(liste_jour),inplace=True)
        
    for o in station.columns:
        station[o]=pd.to_timedelta(station[o])
        #station[o]=station[o].fillna(dt.timedelta(0))
    
    return station
    
    

###################### Données statiques ##########################
            
@st.cache
def load_pkg(nrows=None):
    data=pd.read_csv('/Users/jeremymaignier/Desktop/UPCITI/donnees/recup_auto/gros_park.csv',sep=',')

    format_date=str('%Y-%m-%d %H:%M:%S')
    data['date']=data['date'].apply(lambda x:datetime.strptime(x,format_date))



    data.set_axis([change_type(o) for o in data.columns], axis=1,inplace=True)

    places=[o for o in data.columns if type(o)==int]

    for o in places:
        data[o]=data[o].astype(str)
            
        
    nbre_places=len(places)
    nbre_dates=data.shape[0]

    #data['taux_occup']=[round((np.count_nonzero(data[places].iloc[o]!=str(0.0))/nbre_places)*100,2) for o in range(nbre_dates)]

    data["jour"]=data['date'].dt.day
    data["jour_nom"] = data['date'].dt.day_name()
    data['mois']=data['date'].dt.month
    data['jour_semaine']=data['date'].apply(lambda x: x.isoweekday())
    data['semaine']=data['date'].dt.week
    data['heure']=data['date'].dt.hour
    data["periode"]=data['heure'].apply(conv_period)

    data.drop(places,axis=1,inplace=True)

    if nrows==None:
        return data
    else :
        return data.head(nrows)
        
        
@st.cache
def load_sta(nrows=None):
    station=pd.read_csv('/Users/jeremymaignier/Desktop/UPCITI/donnees/temps_station/gros_station.csv',sep=',')
   
   
    station.set_index(station.columns[0],inplace=True)
        
    
    for o in station.columns:
        station[o]=pd.to_timedelta(station[o])
        station[o]=station[o].fillna(dt.timedelta(0))
    
    
    #for o in station.columns:
        #station[o]=station[o].apply(lambda x:x.seconds)

    if None:
        return station
        
    else :
        return station.head(nrows)
        
        
######################### GRAPHIQUES  ###################

def graph(DF,FORMAT_X,TITRE,TITRE_color,LEGEND_X,LEGEND_Y,COLOR,PAS=None):
    
    fig, ax = plt.subplots(figsize=(14,11))
    
    ax.axhline(y=DF.values.mean(),color='grey',linestyle='--',label='moyenne')
    ax.axhline(y=90,color='red',linestyle='--',label='90%')
    
    
    ax.set_title(TITRE,color=TITRE_color,fontsize=30)
    ax.set_xlabel(LEGEND_X,fontsize=22)
    ax.set_ylabel(LEGEND_Y,fontsize=22)
    
    plt.legend(loc="best",fontsize=18)
    plt.grid(True)

    x = DF.index[0:1]
   
    y=DF.values[0:1]
    #ax.set_xlim(0, occup.shape[0])
    ax.set_ylim(0,110)
    ax.set_xlim()

    majorFmt = mdates.DateFormatter(FORMAT_X)
    ax.xaxis.set_major_formatter(majorFmt)
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.tick_params(axis='both', which='major',labelsize=18)
    fig.autofmt_xdate()
    ax.set_xlim(DF.index[0],DF.index[-1])

    line, = ax.plot(x, y,color=COLOR,linewidth=2)
    the_plot = st.pyplot(plt,color=COLOR)

    def animate(i):
        line.set_data(x,y)
        the_plot.pyplot(plt,color=COLOR)

    if PAS==None:
        line.set_data(DF.index,DF.values)
        the_plot.pyplot(plt,color=COLOR)
        
    else:
        i=0
        pas=PAS
        while i*pas<DF.shape[0]:
            x=x.append(DF.index[len(x):len(x)+pas])
            y=np.concatenate((y,DF.values[y.shape[0]:y.shape[0]+pas]))
            animate(i)
            time.sleep(0.0001)
            i+=1
            
            

@st.cache
def load_data_EDA():
    path='/Users/jeremymaignier/Desktop/UPCITI/donnees/datas_pour_streamlit/occup_preprocess_17-31_mai.csv'
    data=pd.read_csv('/Users/jeremymaignier/Desktop/UPCITI/donnees/datas_pour_streamlit/occup_preprocess_17-31_mai.csv',sep=',')
    data['date']=pd.to_datetime(data['date'])
    test=data.set_index('date')
    
    return test
    
    
    
    
        
    
        
########################## CORPS DU DASHBOARD ######################

def main():
    st.title("Etude du parking d'ETTELBRUCK (Luxembourg)")
    
    liste_choix=["Introduction","Premières approches", "Data Vizualisation", "Données en temps réelles","Analyse exploratoire","Visualisation Spatiale","Algos de Machine Learning"]
    
    
    MENU=st.sidebar.selectbox("MENU",liste_choix)
    
    if MENU == liste_choix[0]:
        st.sidebar.header(MENU)
        st.write("Le parking est composé de 42 places couvertes par les capteurs mis en place afin de monitorer le trafic.")
        st.write("Au travers d'un jeu de données récupéré sur une dizaine de jours, le but de ce Dashboard est d'effectuer une 1ère étude statistique \"primaire\" sur ce jeu de données.")
        st.write("Après avoir récupéré et structuré les données de manière adéquate, la 1ère étape consiste à effectuer une Analyse Exploratrice afin de déceler les éléments pertinents à étudier, les axes d'études possibles ...")
        st.write("En parallèle s'effectuent des tâches de DataViz afin de se représenter les données plus facilement.")
        st.write("L\'idée à terme serait de pouvoir consolider des modèles descriptifs de qualité afin de les déployer ensuite sur de la prédiction future, amenant à des prises de décision variées et légitimes.")
        if st.button("Tout est clair, passons à la suite"):
            st.success("L'introduction est finie, nous pouvons commencer les premières approches aux données")
        
        
    if MENU == liste_choix[1]:
        st.sidebar.header(MENU)
        st.write("Dans cette partie, nous allons découvrir les données recueillies depuis le 17 Mai, le type de variables que l'on a à disposition et des premières statistiques basiques.")
        st.header("Données brutes")
        
        st.subheader("Pour le taux d'occupation")
        with st.spinner("Récupération des données statiques, cela peut prendre quelques instants ..."):
            temp=load_pkg()
            df=temp.copy()
            st.dataframe(df.set_index('date').iloc[:,[o for o in range(5)]])
            st.success("Chargement effectué !")
        st.write("On voit que l'on dispose de données régulières (toutes les 1/2 minutes) du taux d'occupation du parking, ce qui nous permet d'avoir une vision précise de son évolution.")
        
        
        st.subheader("Pour le temps de stationnement")
        with st.spinner("Récupération des données statiques, cela peut prendre quelques instants ..."):
            temp=load_sta()
            station=temp.copy()
            for o in station.columns:
                station[o]=station[o].apply(lambda x:round(x.seconds/3600))

            st.dataframe(station)
            st.success("Chargement effectué !")
            
        st.write("Ici nous avons par jour et par place, le temps moyen de stationnement des véhicules en Heure.")
        
        st.header("Affichage des statistiques de base")
        
        if st.button("Afficher les stats d'occupation"):
        
            temp=load_pkg()
            pkg=temp.copy()
            fig3 = px.box(pkg, x='jour', y='taux_occup', notched=True,title='boxplot')
            boxplot_chart = st.plotly_chart(fig3)
                
            
        if st.button("Afficher les stats de stationnement"):
            temp=load_sta()
            sta=temp.copy()
            st.write(sta.describe().iloc[1:,].apply(lambda x:pd.to_timedelta(x)))
            
            
    if MENU == liste_choix[2]:
        st.sidebar.header(MENU)
        st.write("Parce qu'une image vaut mille mots, nous allons ici se représenter les deux variables graphiquement.")
        affiche=st.sidebar.selectbox("Data Viz",['Occupation','Stationnement'])
        
        
        if affiche=='Occupation':
            st.header(affiche)
            temp=load_pkg()
            parking=temp.copy()
            parking["prec"]=parking["date"].apply(lambda x :datetime.strftime(x,"%y-%m-%d %H:%M"))
            
            
            
           
            semaine = st.selectbox("Quelle semaine(s) vous intéressent ?", parking['semaine'].unique().tolist())
            if not semaine:
                    st.sidebar.error("Vous devez choisir au moins une semaine.")
                        
            temp=parking.set_index('semaine').loc[semaine]
            
            etude="semaine du "+str(datetime.strftime(temp['date'].iloc[0],'%d/%m/%y'))+" au "+str(datetime.strftime(temp.tail(1)['date'].iloc[0],'%d/%m/%y'))
                
            occup=temp.groupby(["prec"],sort=False).agg({"taux_occup":'max'})
            occup.reset_index(inplace=True)
            occup['prec']=occup['prec'].apply(lambda x:datetime.strptime(x,"%y-%m-%d %H:%M"))
            occup.set_index('prec',inplace=True)
            
                
            with plt.style.context('fivethirtyeight'):
                graph(occup,'%A %d/%m',"Taux d'occupation sur la "+etude+"\n",'olive',"Jour","Taux d'occupation en %", 'brown')
                    
            #if len(semaine)==1:
            """
            ---
            """
            
            periode=st.slider('Jour(s) de la semaine',1,7)
            temp_bis=temp.set_index('jour_semaine').loc[[o for o in range(1,periode+1)]]
            
            debut =temp_bis['jour_nom'].iloc[0]
            fin=temp_bis.tail(1)['jour_nom'].iloc[0]
            
            interval=str(debut)+' --> '+str(fin)
            
            utile=temp_bis.groupby(["prec"],sort=False).agg({"taux_occup":'max'})
            utile.reset_index(inplace=True)
            utile['prec']=utile['prec'].apply(lambda x:datetime.strptime(x,"%y-%m-%d %H:%M"))
            utile.set_index('prec',inplace=True)
            
            with plt.style.context('seaborn-notebook'):
                try:
                    graph(utile,'%A %d/%m',"Taux d'occupation sur la période "+str(interval)+ "\n",'black',"Jour","Taux d'occupation en %", 'darkblue')
                except:
                    st.warning('Nous avons pas les données pour la période que vous avez selectionné')
                    
            st.info("On peut observer une évolution pérodique par jour, ou en tout cas certaines ressemblances entre les jours, ce qui présage de bons signes pour les futurs modèles d'analyse et de prédiction")
            
            st.subheader("Autre façon d'afficher le graph, avantage navigation pour l'utilisateur")
            
            ax2= px.line(utile, x=utile.index, y='taux_occup',
            #template='plotly_dark',
            )
            ax2.update_yaxes(range=[0, 100])
            ts_chart = st.plotly_chart(ax2)
            
            st.subheader("Heatmap de la semaine")
            st.write("Au lieu de visualiser le taux d'occupation comme une courbe, on pourrait le représenter de la façon suivante.")
            
            temp.reset_index(inplace=True)
            
            ask=st.selectbox("Moyenne ou Max ?",['mean','max'])
            agg=temp.groupby(['jour_nom','periode'],sort=False).agg({'taux_occup':ask})
            
            
            values=agg.values.reshape(7,5)
            
           

            fig = go.Figure(data=go.Heatmap(
                    z=values.T,
                    x=['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi','Samedi','Dimanche'],
                    y=['Nuit(22-7)', 'Matin(7-11)', 'Midi(11-14)','Après-midi(14-18)','Soir(17-22)'],
                    colorscale='YlGnBu',
                    colorbar = dict(title="Taux (%)")
                    ))

            fig.update_layout(
                title="Heatmap du taux d'occupation selon le jour et la période de la journée",
                xaxis_title="Jours de la semaine",
                yaxis_title="Période de la journée",
                
                
                )
            st.plotly_chart(fig)

            
            
            
        if affiche=='Stationnement':
            st.header(affiche)
            temp=load_sta()
            station=temp.copy()
            
            
            for o in station.columns:
                station[o]=station[o].apply(lambda x:round(x.seconds/3600))
            moy_par_place=temp.mean().apply(lambda x:x.round('1s'))

            moy_par_jour=temp.T.mean()
            moy_gen=temp.mean().mean()
            
            stats=moy_par_jour.reset_index()
            stats.set_axis(['jour','moy'],axis=1,inplace=True)
            stats['jour']=stats['jour'].apply(lambda x:datetime.strptime(x,'%y/%m/%d'))
            stats['moy']=stats['moy'].apply(lambda x:x.seconds//60)
            stats['size']=stats['moy'].apply(lambda x: 10 if (x<=30) else 50 if ((x>=30) and (x<60)) else 100)
            
            with plt.style.context('seaborn'):
                fig1, ax1 = plt.subplots(figsize=(14,10))

                majorFmt = mdates.DateFormatter('%a %d/%m')
                ax1.xaxis.set_major_formatter(majorFmt)
                
                ax1.xaxis.set_major_locator(ticker.AutoLocator())
                ax1.set_xlim(left=stats['jour'][0]-dt.timedelta(1),right=stats['jour'][stats.shape[0]-1]+dt.timedelta(1))
                ax1.xaxis.set_minor_locator(ticker.LinearLocator())
                ax1.tick_params(axis='both', which='major', labelsize=15)


                sns.set(style="darkgrid")
                palette = sns.color_palette("magma_r", n_colors=len(stats['size'].unique()))
                sns.scatterplot(x=stats['jour'], y=stats['moy'],size=stats['size'],hue=stats['size'],palette=palette,sizes=(100,500))


                plt.legend('')
                plt.title("Temps moyen de stationnement par jour \n", fontsize = 25,color='darkslategray')
                plt.xlabel("\nJour",fontsize=20)
                plt.ylabel('Temps moyen de stationnement (minutes)',fontsize=20)
                fig1.autofmt_xdate()
                st.pyplot(plt)
                
                st.info("Les jours pour lesquels les véhicules restent (en moyenne) stationnés plus d'une heure sur le parking apparaissent directement à la vue du lecteur.")
            
            #st.subheader("Vision d'ensemble du parking en terme de temps de stationnement")
            
            heatmap=go.Heatmap(
                    z=station,
                    x=[o for o in range(1,len(station.columns)+1)],
                    y=station.index,
                    colorscale='YlGnBu',
                    colorbar = dict(title="<i>Temps (heure)</i>")
                    )
            layout=go.Layout(
                    title=dict(text="<b>Heatmap des 42 places de parking en fonction des jours</b>",
                            x=0.5,
                            y=0.9,
                            font=dict(color='black',
                                size=15)),
                    xaxis=dict(title='<i>Places</i>'),
                    yaxis=dict(title='<i>Jours</i>'),
                            )
            fig2 = go.Figure(data=[heatmap],layout=layout)
            fig2.update_xaxes(dtick=5)
            fig2.update_yaxes(dtick=2)
            st.plotly_chart(fig2)
            
            
            st.warning("Cette façon de représenter le temps de stationnement par jour et par place semble donner plus de nuances en terme de temps de stationnement que le graphique précédent")
            st.info("Globalement, une grosse majorité des places du parking n'est occupée que maximum une heure")
            
            #data=[station.iloc[o].values for o in range(station.shape[0])]
            
            

    
    if MENU == liste_choix[3]:
        st.sidebar.header(MENU)
        st.write("Il est maintenant possible d'aller récupérer de façon répétée les données qui s'accumulent au fur et à mesure de la journée pour avoir une idée de l'évolution en \"quasi temps réel\".")
        
        temp=load_data_brute().copy()
        pkg=process_pkg(temp)
        sta=process_sta(temp)
        
        if st.button("(Re)lancer les données"):
            with st.spinner("On vide le cache et on récupère les données actualisées, cela peut prendre quelques instants..."):
    
                #caching.clear_cache()
                #temp=load_data_brute()
                #pkg=process_pkg(temp)
                st.dataframe(pkg)
                #sta=process_sta(temp)
                st.dataframe(sta)
                
                occup=pkg[['date','taux_occup']]
                occup.set_index('date',inplace=True)
                

        
            
            trace0=go.Scatter(
                    x=occup.index,
                    y=occup.taux_occup,
                    mode='lines',
                    name='flux',
                    line_shape="spline",
                    line=dict(color='darkblue',width=1.5))
            trace1=go.Scatter(
                    x=occup.index,
                    y=[90 for o in occup.index],
                    mode='lines',
                    name='90%',
                    line=dict(color='brown',dash='dot'))
            trace2=go.Scatter(
            x=occup.index,
            y=[occup.taux_occup.mean() for o in occup.index],
            mode='lines',
            name='moyenne',
            line=dict(color='olive',dash='dash'))
                    
            data=[trace0,trace1,trace2]
            layout=go.Layout(
                    #grid=dict(showgrid=False),
                    title=dict(text="Taux d'occupation en temps réel",x=0.5,y=0.9,
                    font=dict(color='brown'),
                    ),
                    xaxis=dict(title='Jour(s)',
                                #showgrid=True,
                                gridcolor= "lightSteelBlue"
                                ),
                    yaxis=dict(title='Taux d\'occupation (%)',
                                #showgrid=True,
                                gridcolor="lightSteelBlue"
                                ),
                    plot_bgcolor='lavender',
                    paper_bgcolor='lavender',
                    #template='plotly_dark',
                    )
            
            fig=go.Figure(data=data,layout=layout)
            fig.update_yaxes(range=[0, 110])
            fig.update_layout(
            #xaxis_tickformat = '%a %d/%m'
            )
            st_fig=st.plotly_chart(fig)
            
            #ax3= px.line(occup, x=occup.index, y='taux_occup')
            #ts_chart = st.plotly_chart(ax3)
            
        if st.button('Sauvegarder'):
            with st.spinner("Sauvegarde en cours veuillez patienter un instant"):
                path_occup='/Users/jeremymaignier/Desktop/UPCITI/donnees/collecte_url/occupation'
                path_temps='/Users/jeremymaignier/Desktop/UPCITI/donnees/collecte_url/temps'
                name_occup=os.path.join(path_occup,pkg.date.dt.strftime("%d_%m_%Hh")[0]+'au'+pkg.date.dt.strftime("%d_%m_%Hh").iloc[-1]+'.csv')
                pkg.to_csv(name_occup,header=True,index=False)
                
                name_temps=os.path.join(path_temps,sta.index.strftime("%d_%m")[0]+'au'+sta.index.strftime("%d_%m")[-1]+'.csv')
                sta.to_csv(name_temps,header=True,index=False)
                
                st.success("Sauvegarde effectuée avec succès !")
            
            
            
            
            
            
        
        
    if MENU == liste_choix[4]:
        st.sidebar.header(MENU)
        st.write("Maintenant que nous avons pu exploiter en surface les données, tentons d'apporter un expertise plus poussée.")
        st.write("Nous allons d'abord pratiquer une analyse exploratrice, basée sur l'étude des corrélations possibles. Il s'agit de deux problèmes de régression.")
        
        """
        ---
        """
        
        data=load_data_EDA().copy()
        st.write(data)
        """
        ---
        """
        liste_graphe=["Vue d'ensemble","Pairplot","Heatmap","Visuel de la distribution"]
        graphe=st.sidebar.selectbox("""GRAPHE""",liste_graphe+["Stationnement"])
        
        if graphe==liste_graphe[0]:
            data2=pd.read_csv("/Users/jeremymaignier/Desktop/UPCITI/donnees/preprocess/premier_dataset_preprocess.csv",index_col=0)
            st.write(data2)
            st.info("""Taux d'occupation par jour selon les 3 périodes de la journée, la suite est une vision globale de la distribution de chaque variable""")
            """
            ---
            """
            st.header("Boxplots du taux en fonction des variables qualitatives")
            
            fig = make_subplots(rows=1, cols=3,shared_yaxes=True)
            fig.add_trace(go.Box(x=data2['opinion'],y=data2['taux_occup'],boxmean=True,boxpoints='outliers',name='Météo'),row=1,col=1)
            fig.add_trace(go.Box(x=data2['periode'],y=data2['taux_occup'],boxmean=True,boxpoints='outliers',name='Période de la journée'),row=1,col=2)
            
            fig.add_trace(go.Box(x=data2['type_jour'],y=data2['taux_occup'],boxmean=True,boxpoints='outliers',name='Jour de la semaine'),row=1,col=3)
            
            fig.update_layout(height=600,width=700)

            
            st.plotly_chart(fig)
            st.success(""" Le Taux est :
            \n- plus important le Midi que le Soir ou Matin
            \n- plus dispersé en jour normal qu'en week-end ou jour férié
            \n- plus dispersé en temps de bonne météo """)
            
            """
            ---
            """
            liste_choix_relplot=[o for o in data2.columns[2:6]]
            choix_relplot=st.selectbox("Quelle variable quantitative ?",liste_choix_relplot)
            
            relplot=sns.relplot(data=data2,x=choix_relplot,y='taux_occup',col='periode',row='type_jour',hue='opinion')
            st.pyplot(relplot)
            
            """
            ---
            """
            st.header("Pairplot d'ensemble")
            
            liste_choix_hue=["Période de la journée","Jour de la semaine","Météo"]
            choix_hue=st.selectbox("Selon quelle variable qualitative voulez-vous voir les distributions ?",liste_choix_hue)
            
            if choix_hue==liste_choix_hue[0]:
                st.pyplot(sns.pairplot(data2,hue='periode'))
                st.info("""La distribution est significativement différente selon la période de la journée (1er graphe)""")
                
            if choix_hue==liste_choix_hue[1]:
                st.pyplot(sns.pairplot(data2,hue='type_jour'))
            
            if choix_hue==liste_choix_hue[2]:
                st.pyplot(sns.pairplot(data2,hue='opinion'))
            """
            ---
            """
            
            st.header("Distribution du taux en fonction des variables quantitatives")
            

            fig2,axes2=plt.subplots(2,2,figsize=(20,16))
            
            sns.regplot(x=data2.columns[2],y='taux_occup',data=data2,ax=axes2[0,0])
            #axes[1,0].set_xlabel(data2.columns[2])
            sns.regplot(x=data2.columns[3],y='taux_occup',data=data2,ax=axes2[0,1])
            axes2[0,1].set_xlabel(data2.columns[3])
            sns.regplot(x=data2.columns[4],y='taux_occup',data=data2,ax=axes2[1,0])
            axes2[1,0].set_xlabel(data2.columns[4])
            sns.regplot(x=data2.columns[5],y='taux_occup',data=data2,ax=axes2[1,1])
            axes2[1,1].set_xlabel(data2.columns[5])
            st.pyplot(fig2)
            st.success("""Il semblerait que le taux :
             \n- augmente légèrement avec la température et le vent
             \n- baisse légèrement avec l'humidité
             \n- NB : interprétations à creuser car trop peu de données encore""")
            """
            ---
            """
          
            st.header("Distribution des variables quantitatives")
            fig3,axes3=plt.subplots(5,1,figsize=(15,20))
            
            sns.distplot(data2.taux_occup,bins=40,ax=axes3[0])
            sns.distplot(data2['temperatures(C)'],bins=40,ax=axes3[1])
            sns.distplot(data2['wind(KMH)'],bins=40,ax=axes3[2])
            sns.distplot(data2['precip(MM)'],bins=40,ax=axes3[3])
            sns.distplot(data2["humid(%)"],bins=40,ax=axes3[4])
            
            st.pyplot(fig3)
            
            st.success("""- Le taux d'occupation semble suivre une loi normale centrée autour de 60""")
            
            """
            ---
            """
            st.header("Matrice des corrélations")
        
            fig,axe=plt.subplots(figsize=(12,7))
            
            sns.heatmap(data2.corr(), annot = True,annot_kws={'size':15},ax=axe,vmin=-1,vmax=1,center=0,linewidths=0.1,fmt='.2g')
            axe.set_ylim([0,5])
            plt.yticks(rotation=0)
            st.pyplot(fig)
            
            st.info("""Le coefficient de corrélation varie de -1 à 1:
            \n- Plus la valeur se rapproche de 1, plus les deux variables sont corrélées positivement,c'est-à-dire qu'elles évoluent dans la même dynamique.
            \n- A l'inverse, plus cette valeur se rapproche de -1, plus elles évoluent dans des dynamiques contraires.
            \n- Enfin, si cette valeur est proche de 0, les deux variables sont plutôt indépendantes, elles ont des dynamiques qui évoluent sans vraiment de lien.""")
            
        
            
        if graphe==liste_graphe[1]:
            st.subheader("Corrélation selon la météo")
            st.write("Ce graphique nous permet d'avoir une vision de la distribution des points selon chacune des variables à disposition.")
            pairplot=sns.pairplot(data,hue='OPINION')
            st.pyplot(pairplot)
        
        
            st.info("La première ligne nous intéresse particulièrement car elle correspond à la distribution du taux d'occupation selon toutes les variables explicatives.")
            st.info("""On voit déjà :\n - Graphe 1: Le taux est significativement différent selon le type de météo \n""")
            """
            ---
            """
            st.subheader("Corrélation selon le type de jour")
            st.write("Voyons si il y une différence significative selon s'il s'agit d'un jour de la semaine lambda, ou bien d'un jour férié, d'un week-end")
        
            pairplot_bis=sns.pairplot(data,hue='type_jour')
            st.pyplot(pairplot_bis)
            st.info(""" A nouveau, la distribution du taux d'occupation semble bien différente entre les jours de week-end et les autres jours de la semaine.""")
            
            st.warning("Les graphiques de la diagonale sont ceux qui nous importent tout particulièrement car ils permettent d'avoir une rapide première vision d'ensemble sur les distributions des différentes variables")
        
        if graphe==liste_graphe[2]:
            fig = go.Figure(data=go.Heatmap(
            x=data.corr().columns,
            y=data.corr().columns,
            z=data.corr().values,
            #y=data.corr().columns,
            #x=data.corr().columns,
            #y=data.corr().columns,
            
            colorbar=dict(tickvals=[-1,-0.5,0, 0.5,1]),
            colorscale='magma',
            ))
            st.plotly_chart(fig)
            
            st.info("A retenir de ce graph :\n - plus il fait chaud, plus le taux baisse\n - le taux est peu corrélé au vent, à la pluie, ou à l'humidité")
            
        if graphe==liste_graphe[3]:
            #var=st.sidebar.selectbox("Taux en fonction de :",[data.columns[o] for o in [1,2,3,5]])
            #st.write(data.columns[[1,2,3,5]].tolist())
            fig,ax1=plt.subplots(2,2,figsize=(11,9))
            
            sns.jointplot(y='taux_occup',
            x=data.columns[1],data=data,
            kind='reg',ax=ax1[0,0])
            ax1[0,0].set_xlabel(data.columns[1])
            
            sns.jointplot(y='taux_occup',
            x=data.columns[2],data=data,
            kind='reg',ax=ax1[0,1])
            ax1[0,1].set_xlabel(data.columns[2])
            
            sns.jointplot(y='taux_occup',
            x=data.columns[3],data=data,
            kind='reg',ax=ax1[1,0])
            ax1[1,0].set_xlabel(data.columns[3])
            
            sns.jointplot(y='taux_occup',
            x=data.columns[5],data=data,
            kind='reg',ax=ax1[1,1])
            ax1[1,1].set_xlabel(data.columns[5])
            
            
            ax1[0,0].set_ylabel('taux_occup')
            ax1[0,1].set_ylabel('taux_occup')
            ax1[1,0].set_ylabel('taux_occup')
            ax1[1,1].set_ylabel('taux_occup')
        
            st.pyplot(fig)
            st.info("""Ceci est une visualisation du taux selon les variables de VENT, PLUIE, HUMIDITE, TEMPERATURE.
            \nIl ne semble pas vraiment y avoir de tendance (plus ou moins des lignes horizontales), sauf quand même pour la température.
            \nAinsi, on pourrait penser que plus la température est forte, moins le parking semble occupé.""")
            
            """
            ---
            """
            
            wknd=data.loc[data.type_jour=='week-end']
            normal=data.loc[data.type_jour=='normal']
            ferie=data.loc[data.type_jour=='férié']
            #st.write(wknd)
            fig2,axes2=plt.subplots(2,2,figsize=(12,9))
            #sns.kdeplot(wknd[wknd.columns[1]],wknd['taux_occup'],ax=axes2[0,0],cmap='Blues')
            
            #sns.kdeplot(normal[normal.columns[1]],normal['taux_occup'],ax=axes2[0,0],cmap='Reds')
            
            #sns.kdeplot(ferie[ferie.columns[1]],ferie['taux_occup'],ax=axes2[0,0],cmap='Greens')
            
            
            
            
            #axes2[0,0]=sns.kdeplot(wknd[wknd.columns[1]],wknd.taux_occup,cmap='Reds')
            sns.jointplot(y='taux_occup',
            x=data.columns[1],data=data,
            kind="kde",ax=axes2[0,0])
            axes2[0,0].set_xlabel(data.columns[1])
            
            sns.jointplot(y='taux_occup',
            x=data.columns[2],data=data,
            kind="kde",ax=axes2[0,1])
            

            axes2[0,1].set_xlabel(data.columns[2])
                
            sns.jointplot(y='taux_occup',
                x=data.columns[3],data=data,
                kind="kde",ax=axes2[1,0])
            axes2[1,0].set_xlabel(data.columns[3])
                
            sns.jointplot(y='taux_occup',
                x=data.columns[5],data=data,
                kind="kde",ax=axes2[1,1])
            axes2[1,1].set_xlabel(data.columns[5])
                
                
            axes2[0,0].set_ylabel('taux_occup')
            axes2[0,1].set_ylabel('taux_occup')
            axes2[1,0].set_ylabel('taux_occup')
            axes2[1,1].set_ylabel('taux_occup')
            
            st.pyplot(fig2)
            st.info("""Ceci permet de se rendre compte de la disposition de nos données.
            \nLes zones les plus sombres sont celles où les données sont le plus présentes. Cela fait donc ressortir les zones denses de nos données, avec un contour approximatif de la zone de distribution.""")
            
            #st.pyplot(fig2)
            
            wknd=data.loc[data.type_jour=='week-end']
            normal=data.loc[data.type_jour=='normal']
            ferie=data.loc[data.type_jour=='férié']
            
            
            figure,a=plt.subplots()
            a=sns.kdeplot(wknd[wknd.columns[1]],wknd.taux_occup,cmap='Reds')
            a.text(20.0,40,"week-end",color='red')
            a=sns.kdeplot(normal[normal.columns[1]],normal.taux_occup,cmap='Greens')
            a.text(12.5,90,"jour normal",color='green')
            
            st.pyplot(figure)
            st.info("""Ici c'est le même principe que le graphe précédent, à ceci près qu'il différencie la distribution selon le type de jour (pas assez de données encore pour se faire une véritable idée)""")
            #ax = sns.kdeplot(wknd, setosa.sepal_length,
               #              cmap="Reds", shade=True, shade_lowest=False)
           # ax = sns.kdeplot(virginica.sepal_width, virginica.sepal_length,
                  #            cmap="Blues", shade=True, shade_lowest=False)
                  
                  
                  
        if graphe=="Stationnement":
            sta=pd.read_csv('/Users/jeremymaignier/Desktop/UPCITI/donnees/preprocess/test_22_06_type_jour+tps_seconds_pres.csv',index_col=0)
            st.write(sta)
            
            fig = make_subplots(rows=1,cols=2,shared_yaxes=True)
            fig.add_trace(go.Box(x=sta['OPINION'],y=sta['tps'],boxmean=True,boxpoints='outliers',name='Météo'),row=1,col=1)
            fig.add_trace(go.Box(x=sta['type_jour'],y=sta['tps'],boxmean=True,boxpoints='outliers',name='Jour de la semaine'),row=1,col=2)
            
            
            fig.update_layout(height=600,width=700)
            st.plotly_chart(fig)
            """
            ___
            """
            liste_choix_relplot=[o for o in sta.columns[[1,2,3,5]]]
            choix_relplot=st.selectbox("Quelle variable quantitative ?",liste_choix_relplot)
            
            relplot=sns.relplot(data=sta,x=choix_relplot,y='tps',col='OPINION',row='type_jour')
            st.pyplot(relplot)
            """
            ---
            """
            liste_choix_hue=["Jour de la semaine","Météo"]
            choix_hue=st.selectbox("Selon quelle variable qualitative voulez-vous voir les distributions ?",liste_choix_hue)
            
            if choix_hue==liste_choix_hue[0]:
                st.pyplot(sns.pairplot(sta,hue='type_jour'))
                
            if choix_hue==liste_choix_hue[1]:
                st.pyplot(sns.pairplot(sta,hue='OPINION'))
            """
            ---
            """
            fig2,axes2=plt.subplots(2,2,figsize=(20,16))
            
            sns.regplot(x=sta.columns[1],y='tps',data=sta,ax=axes2[0,0])
            axes2[0,0].set_xlabel(sta.columns[1])
            sns.regplot(x=sta.columns[2],y='tps',data=sta,ax=axes2[0,1])
            axes2[0,1].set_xlabel(sta.columns[2])
            sns.regplot(x=sta.columns[3],y='tps',data=sta,ax=axes2[1,0])
            axes2[1,0].set_xlabel(sta.columns[3])
            sns.regplot(x=sta.columns[5],y='tps',data=sta,ax=axes2[1,1])
            axes2[1,1].set_xlabel(sta.columns[5])
            st.pyplot(fig2)
            """
            ---
            """
            fig,axe=plt.subplots(figsize=(12,7))
            
            sns.heatmap(sta.corr(), annot = True,annot_kws={'size':15},ax=axe,vmin=-1,vmax=1,center=0,linewidths=0.1,fmt='.2g')
            axe.set_ylim([0,5])
            plt.yticks(rotation=0)
            st.pyplot(fig)
            
            
    if MENU== liste_choix[5]:
        st.title(MENU)
        
        
        
        
        
    if MENU==liste_choix[6]:
        st.title(MENU)
        liste_ML=['Clustering','Regression','Classification']
        ML=st.sidebar.selectbox("Machine Learning",liste_ML)
        
        
        if ML==liste_ML[0]:
            st.header(ML)
            st.write("Nous allons ici étudier les similitudes entre certaines données pour établir des groupes d'individues qui auraient les mêmes caractéristiques.")
            st.write("Il est possible de considérer comme \"individus\" ici, soit les 7 jours de la semaine, soit les 42 places.")
            
            liste_algos_clus=['ACP/K-means']
            choix_clus=st.sidebar.selectbox("Quel algo ?",liste_algos_clus)
            
            if choix_clus==liste_algos_clus[0]:
                st.subheader("Analyse en Composantes Principales")
                st.write("Afin de réduire la dimensionalité et ainsi de se représenter les données sur un plan en 2D ou à la limite 3D, nous effectuons une ACP qui va permettre \"d'aplatir\" au mieux le jeu de données, sans pour autant perdre en chemin de l'information.")
                st.write("Nous allons nous pencher ici sur les ressemblances possibles entre certains jours, c'est-à-dire les jours qui ont des caractéristiques similaires du point de vue du temps de stationnement du parking. ")
                from sklearn.preprocessing import StandardScaler
                from sklearn.decomposition import PCA
                from sklearn.cluster import KMeans
                
                temp=load_sta().copy()
                #jour=temp.to_numpy()
                test=pd.concat([temp,pd.DataFrame(temp.mean(axis=1),columns=['mean']),pd.DataFrame(temp.std(axis=1),columns=['std'])],axis=1)
                temp=test
                jour=temp.to_numpy()
                places=temp.T.to_numpy()
                scaler=StandardScaler()
                jour_scaled=scaler.fit_transform(jour)
                places_scaled=scaler.fit_transform(places)
                
                liste_dim=["2D","3D"]
                dim_choix=st.selectbox('Quelle représentation ?',liste_dim)
                
                
                
                if dim_choix=="2D":
                    st.write(test)
                    st.write(temp.std(axis=1))
                    pca2D=PCA(n_components=2)
                    pca2D_bis=PCA(n_components=2)
                    ##########
                    coord_jour=pca2D.fit_transform(jour_scaled)
                    clus_jour = KMeans(n_clusters=3, random_state=0).fit(coord_jour)
                    JOUR=pd.concat([pd.DataFrame(temp.index),pd.DataFrame(coord_jour),pd.DataFrame(clus_jour.predict(coord_jour))],axis=1)
                    
                    
                    #JOUR.set_axis(temp.index,axis='index',inplace=True)
                    JOUR.set_axis(['date','coord1','coord2','classe'],axis='columns',inplace=True)
                    JOUR['date']=pd.to_datetime(JOUR['date'],format='%y/%m/%d')
                    
                    #st.write(JOUR)
                    #tout=go.Scatter(
                       #     x=JOUR.iloc[:,0],
                       #     y=JOUR.iloc[:,1],
                       #     text=JOUR.index,
                        #    mode='markers',
                        #    name='',
                        #    showlegend=False,
                            
                         #   marker=dict(size=7,color=JOUR["classe"])
                            #)
                    
                    #figure=go.Figure(data=tout)
                   # st.plotly_chart(figure)
                    
                #########
                    cluster_jour_1=JOUR.loc[JOUR['classe'] == 0]
                    cluster_jour_2=JOUR.loc[JOUR['classe'] == 1]
                    cluster_jour_3=JOUR.loc[JOUR['classe'] == 2]

                    scatter1 = go.Scatter(
                        mode = "markers",
                        name = "Cluster 1",
                        text=cluster_jour_1['date'].dt.strftime("%a %d %B"),
                        x = cluster_jour_1.iloc[:,1],
                        y = cluster_jour_1.iloc[:,2],
                            
                        
                        marker = dict(size=9, color='green')
                    )
                    scatter2 = go.Scatter(
                        mode = "markers",
                        name = "Cluster 2",
                        text = cluster_jour_2['date'].dt.strftime("%a %d %B"),
                        
                        x = cluster_jour_2.iloc[:,1],
                        y = cluster_jour_2.iloc[:,2],
                        marker = dict( size=9, color='blue')
                    )
                    scatter3 = go.Scatter(
                        mode = "markers",
                        name = "Cluster 3",
                        text=cluster_jour_3['date'].dt.strftime("%a %d %B"),
                        x = cluster_jour_3.iloc[:,1],
                        y = cluster_jour_3.iloc[:,2],
                        marker = dict(size=9, color='red')
                    )
                    centres_jours = go.Scatter(
                            x=clus_jour.cluster_centers_[:,0],
                            y=clus_jour.cluster_centers_[:,1],
                            mode='markers',
                            name='Centres',
                            marker = dict(size=5, color='black'))

                    
                    data_jours= [scatter1,scatter2,scatter3,centres_jours]
                    #### LAYOUTS
                    layout_jours = go.Layout(
                    title = dict(text='Clustering en 2 dimensions des 11 jours',
                            x=0.5,
                            y=0.9),
                    xaxis=dict(showgrid=False,linecolor='black'#gridcolor="lightSteelBlue"
                    ),
                    yaxis=dict(showgrid=False,
                    linecolor='black',
                    #gridcolor='lightSteelBlue'
                    ),
                    plot_bgcolor='lavender',
                    paper_bgcolor='lavender',)
                    
                    ### AFFICHAGE
                    fig = dict( data=data_jours, layout=layout_jours )
                    st.plotly_chart(fig)
                    
                    
                    st.subheader("Aide à l'interprétation")
                    st.write('Nous avons projetté les individus sur 2 axes, mais à quoi peuvent bien correspondre ces deux axes ?')
                    
                
                
                    
                    """
                    ---
                    """
                    
                    nbre_clusters=st.slider("Combien de Clusters ?",1,places.shape[0],3)
                    coord_places=pca2D_bis.fit_transform(places_scaled)
                    
                    clus_places=KMeans(n_clusters=nbre_clusters, random_state=0).fit(coord_places)
                    
                    st.write("L'inertie correspondant à {} clusters est ".format(nbre_clusters),clus_places.inertia_)
                    
                    
                    #st.write("Variance expliquée est ",sum(pca2D_bis.explained_variance_ratio_))
                    
                    
                    eigval=(2-1)/2*pca2D.explained_variance_
                    sqrt_eigval = np.sqrt(eigval)
                    

                    corvar = np.zeros((44,2))
                    for k in range(2):
                        corvar[:,k] = pca2D.components_[k,:]*sqrt_eigval[k]
                   
                                       
                    SCAT = go.Scatter(
                    mode = "markers",
                    name = "test 1",
                    text=temp.columns,
                    
                    
                    x= corvar[:,0],
                    y = corvar[:,1],
                                               
                                           
                    marker = dict(size=9, color='green')
                                       )
                                       
                    shapes=[{"type":"line","x0":0,"y0":0,"x1":o,"y1":p} for o,p in zip(corvar[:,0],corvar[:,1])]
                    FIG=go.Figure(data=SCAT)
                    FIG.add_shape(type="circle",
                    #fillcolor="blue",
                        x0=-1,
                                       y0=-1,
                                       x1=1,
                                       y1=1,
                                       line_color="blue")
                    FIG.update_layout(dict(shapes=shapes))
                    st.plotly_chart(FIG)
                                       
                    
                    ### SCATTERS
                    fast= go.Scatter(
                        x=[o for o in range(1,places.shape[0]+1)],
                        y=[KMeans(n_clusters=o, random_state=0).fit(coord_places).inertia_ for o in range(1,places.shape[0]+1)],
                        showlegend=False)
                        
                    point=go.Scatter(
                        x=[nbre_clusters],
                        y=[KMeans(n_clusters=nbre_clusters, random_state=0).fit(coord_places).inertia_],
                        mode='markers',
                        marker=dict(size=9),
                        showlegend=False)
                    
                    ### LAYOUTS
                    layout_fast= go.Layout(
                        title = dict(text='Inertie en fonction du nombre de clusters',
                        x=0.5,
                        y=0.9),
                        xaxis=dict(title="Nombre de Clusters désiré"),
                        yaxis=dict(title="Inertie totale"),
                        )
                    
                    ### AFFICHAGE
                    fig_fast=go.Figure(data=[fast,point], layout=layout_fast)
                    st.plotly_chart(fig_fast)
                
                    
                    
                    """
                    ---
                    """
                    
                    
            
                    
                    
                    nbre_choisi=st.number_input("Choisissez le nombre de clusters que vous voulez",1,places.shape[0],1)
                    clus_choisi=KMeans(n_clusters=nbre_choisi,random_state=0).fit(coord_places)
                    PLACES=pd.concat([pd.DataFrame(coord_places),pd.DataFrame(clus_choisi.predict(coord_places))],axis=1)
                    PLACES.set_axis(temp.columns,axis='index',inplace=True)
                    PLACES.set_axis(['coord1','coord2','classe'],axis='columns',inplace=True)
                    
                    if st.button("Afficher le Clustering"):
                        color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(nbre_choisi)]
                        
                        
                        ### SCATTERS
                        trace_places=[]
                        
                        for i in range(nbre_choisi):
                            places_temp=PLACES.loc[PLACES['classe'] == i]
                            x_temp=places_temp.iloc[:,0]
                            y_temp=places_temp.iloc[:,1]
                            scatter1_temp = go.Scatter(
                                x = x_temp,
                                y = y_temp,
                                mode = "markers",
                                name = "Classe {}".format(i+1),
                                text=str('Place ')+places_temp.index,
                                marker = dict(size=9, color=color[i]))
                            trace_places.append(scatter1_temp)
                        
                       
                        
                    
                        centres_places = go.Scatter(
                            x=clus_choisi.cluster_centers_[:,0],
                            y=clus_choisi.cluster_centers_[:,1],
                            mode='markers',
                            name='Centres',
                            marker = dict(size=5, color='black'))


                        data_places=trace_places+[centres_places]
                        
                        
                        ### LAYOUTS
                        layout_places = go.Layout(
                        title = dict(text='Clustering en 2 dimensions des 42 places',
                                    x=0.5,
                                    y=0.9),
                       #showlegend=False,
                        xaxis=dict(showgrid=False,linecolor='black'#gridcolor="lightSteelBlue"
                                ),
                        yaxis=dict(showgrid=False,
                                linecolor='black',
                                #gridcolor='lightSteelBlue'
                                ),
                        plot_bgcolor='lightSteelBlue',
                        paper_bgcolor='lightSteelBlue',)
                        
                        
                        ### AFFICHAGE
                        fig = go.Figure(data=data_places, layout=layout_places )
                        
                        st.plotly_chart(fig)

                    
                    
                if dim_choix=="3D":
                    st.write(places)
                    pca3D=PCA(n_components=3)
                    pca3D_bis=PCA(n_components=3)
                        
                    coord_jour_3D=pca3D.fit_transform(jour_scaled)
                    coord_places_3D=pca3D_bis.fit_transform(places_scaled)
                    
                    
                    clus_jour_3D = KMeans(n_clusters=3, random_state=0).fit(coord_jour_3D)
                    JOUR_3D=pd.concat([pd.DataFrame(temp.index),pd.DataFrame(coord_jour_3D),pd.DataFrame(clus_jour_3D.predict(coord_jour_3D))],axis=1)
                    #JOUR_3D.set_axis(temp.index,axis='index',inplace=True)
                    JOUR_3D.set_axis(['date','coord1','coord2','coord3','classe'],axis='columns',inplace=True)
                    
                    JOUR_3D['date']=pd.to_datetime(JOUR_3D['date'],format='%y/%m/%d')
                    
                    
                    trace_jours_3D=[]
                    color3D = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(3)]
                    for i in range(3):
                        jour3D_temp=JOUR_3D.loc[JOUR_3D['classe'] == i]
                        x_temp=jour3D_temp.iloc[:,1]
                        y_temp=jour3D_temp.iloc[:,2]
                        z_temp=jour3D_temp.iloc[:,3]
                        scatter3D_temp = go.Scatter3d(
                            x = x_temp,
                            y = y_temp,
                            z=z_temp,
                        
                            mode = "markers",
                            name = "Classe {}".format(i+1),
                                text=jour3D_temp['date'].dt.strftime("%a %d %B"),
                                marker = dict(size=9, color=color3D[i]))
                        trace_jours_3D.append(scatter3D_temp)
                        
                        
                    centres3D_jours = go.Scatter3d(
                        x=clus_jour_3D.cluster_centers_[:,0],
                        y=clus_jour_3D.cluster_centers_[:,1],
                        z=clus_jour_3D.cluster_centers_[:,2],
                        mode='markers',
                        name='Centres',
                        marker = dict(size=5, color='black'))

                    
                    data3D_jours= trace_jours_3D+[centres3D_jours]
                    #### LAYOUTS
                    layout3D_jours = go.Layout(
                    title = dict(text='<b>Clustering en 3 dimensions des 11 jours</b>',
                            x=0.5,
                            y=0.9),
                    xaxis=dict(showgrid=False,linecolor='black'#gridcolor="lightSteelBlue"
                    ),
                    yaxis=dict(showgrid=False,
                    linecolor='black',
                    #gridcolor='lightSteelBlue'
                    ),
                    plot_bgcolor='lavender',
                    paper_bgcolor='lavender',
                    margin=dict(l=10, r=10, t=10, b=10))
                    
                    ### AFFICHAGE
                    fig3D_jours = dict( data=data3D_jours, layout=layout3D_jours )
                    
                    st.plotly_chart(fig3D_jours)
                    
                    st.write("Variance expliquée",sum(pca3D.explained_variance_ratio_))
                    
                    """
                    ___
                    """
                    nbre_clusters_3D=st.slider("Combien de Clusters ?",1,places.shape[0],5)
                        
                        
                    clus_places_3D=KMeans(n_clusters=nbre_clusters_3D, random_state=0).fit(coord_places_3D)
                        
                    st.write("L'inertie correspondant à {} clusters est ".format(nbre_clusters_3D),clus_places_3D.inertia_)
                        
                        
                        
                        ### SCATTERS
                    fast_3D= go.Scatter(
                            x=[o for o in range(1,places.shape[0]+1)],
                            y=[KMeans(n_clusters=o, random_state=0).fit(coord_places_3D).inertia_ for o in range(1,places.shape[0]+1)],
                            showlegend=False)
                            
                    point_3D=go.Scatter(
                            x=[nbre_clusters_3D],
                            y=[KMeans(n_clusters=nbre_clusters_3D, random_state=0).fit(coord_places_3D).inertia_],
                            mode='markers',
                            marker=dict(size=9),
                            showlegend=False)
                        
                        ### LAYOUTS
                    layout_fast_3D= go.Layout(
                            title = dict(text='Inertie en fonction du nombre de clusters en 3D',
                            x=0.5,
                            y=0.9),
                            xaxis=dict(title="Nombre de Clusters désiré"),
                            yaxis=dict(title="Inertie totale"),
                            )
                        
                        ### AFFICHAGE
                    fig_fast_3D=go.Figure(data=[fast_3D,point_3D], layout=layout_fast_3D)
                    st.plotly_chart(fig_fast_3D)
                    
                    
                    
                    
                    
                    
                    nbre_choisi=st.number_input("Choisissez le nombre de clusters que vous voulez",1,places.shape[0],1)
                    
                    
                    
                    clus_choisi_3D=KMeans(n_clusters=nbre_choisi, random_state=0).fit(coord_places_3D)
                    
                    PLACES_3D=pd.concat([pd.DataFrame(coord_places_3D),pd.DataFrame(clus_choisi_3D.predict(coord_places_3D))],axis=1)
                    PLACES_3D.set_axis(temp.columns,axis='index',inplace=True)
                    PLACES_3D.set_axis(['coord1','coord2','coord3','classe'],axis='columns',inplace=True)
                    
                    if st.button("Afficher le Clustering"):
                        color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(nbre_choisi)]
                        
                        ### SCATTERS
                        trace_places_3D=[]
                        
                        for i in range(nbre_choisi):
                            places_temp_3D=PLACES_3D.loc[PLACES_3D['classe'] == i]
                            x_temp=places_temp_3D.iloc[:,0]
                            y_temp=places_temp_3D.iloc[:,1]
                            z_temp=places_temp_3D.iloc[:,2]
                            
                            scatter3D_temp = go.Scatter3d(
                                x = x_temp,
                                y = y_temp,
                                z=z_temp,
                                mode = "markers",
                                name = "Classe {}".format(i+1),
                                text=str('Place ')+places_temp_3D.index,
                                marker = dict(size=9, color=color[i]))
                            trace_places_3D.append(scatter3D_temp)
                        
                       
                        
                    
                        centres3D_places = go.Scatter3d(
                            x=clus_choisi_3D.cluster_centers_[:,0],
                            y=clus_choisi_3D.cluster_centers_[:,1],
                                z=clus_choisi_3D.cluster_centers_[:,2],
                            mode='markers',
                            name='Centres',
                            marker = dict(size=5, color='black'))


                        data3D_places=trace_places_3D+[centres3D_places]
                        
                        
                        ### LAYOUTS
                        layout3D_places = go.Layout(
                        title = dict(text='<b>Clustering en 3 dimensions des 42 places</b>',
                                    x=0.5,
                                    y=0.9),
                       #showlegend=False,
                        xaxis=dict(showgrid=False,linecolor='black'#gridcolor="lightSteelBlue"
                                ),
                        yaxis=dict(showgrid=False,
                                linecolor='black',
                                #gridcolor='lightSteelBlue'
                                ),
                        plot_bgcolor='lightSteelBlue',
                        paper_bgcolor='lightSteelBlue',
                        margin=dict(l=10, r=10, t=10, b=10))
                        
                        
                        ### AFFICHAGE
                        fig3D_places = go.Figure(data=data3D_places, layout=layout3D_places )
                        
                        st.plotly_chart(fig3D_places)
                        
                        quakes = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv')

                        #import plotly.graph_objects as go
                        #fig = go.Figure(go.Densitymapbox(lat=quakes.Latitude, lon=quakes.Longitude, z=quakes.Magnitude,
                                                        # radius=10))
                        #fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=180)
                        #fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                        #st.plotly_chart(fig)

        if ML==liste_ML[1]:
            from sklearn.svm import SVR
            liste_algos_reg=['Support Vector Regressor']
            choix_reg=st.sidebar.selectbox("Quel algo ?",liste_algos_reg)
            if choix_reg==liste_algos_reg[0]:
                st.header(ML)
                data=load_data_EDA().copy()
                st.write(data)
                
                st.write("""On a vu lors de l'analyse exploratrice qu'il pourrait y avoir une relation affine entre la température et le taux.""")
                
                X=data[[data.columns[2]]].to_numpy()
                Y=data[['taux_occup']].to_numpy()
                svr=SVR(kernel='poly',C=100,gamma='scale',degree=3,epsilon=0.1
                )
                svr.fit(X,Y)
                st.write(svr.fit(X,Y).predict(X))
                
                fig,axes=plt.subplots()
                axes.plot(X,svr.fit(X,Y).predict(X),color='r')
                axes.scatter(X[svr.support_], Y[svr.support_])
                st.pyplot(fig)
                from sklearn.metrics import mean_absolute_error
                from sklearn.metrics import mean_squared_error
                st.write(mean_absolute_error(Y,svr.fit(X,Y).predict(X)))
                
                
            
            
                        
                        
                        
                        
                        
                    
                    
                
                    
                    
                
                
                
     
     
        
    
    
    
    
    
    
    
    
    
    
    


if __name__=="__main__":
    main()



# In[5]:



