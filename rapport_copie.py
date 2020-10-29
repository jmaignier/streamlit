#!/usr/bin/env python
# coding: utf-8

# In[1]:

## Pandas packages
import streamlit as st

## Visu
def main():
    st.set_page_config(page_title='Application de test modifiée', page_icon = '🤡', layout = 'wide', initial_sidebar_state = 'auto')
    st.title("ESSAI")
    name = st.text_input("enter your name")
    if name :
        st.success(f"Hello {name}")
    
if __name__=="__main__":
    main()
