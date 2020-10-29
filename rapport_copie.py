#!/usr/bin/env python
# coding: utf-8

# In[1]:

## Pandas packages
import streamlit as st
import matplotlib.pyplot as plt 
import pandas as pd

## Visu
def main():
    st.set_page_config(page_title='Application de test modifiée', page_icon = '🤡', layout = 'wide', initial_sidebar_state = 'auto')
    st.title("ESSAI")
    name = st.text_input("enter your name")
    if name :
        st.success(f"Hello {name}")
        
    fig,ax=plt.subplots()
    plt.plot([1,2],[2,4])
    st.pyplot(fig)
    
if __name__=="__main__":
    main()
