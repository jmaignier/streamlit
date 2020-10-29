#!/usr/bin/env python
# coding: utf-8

# In[1]:

## Pandas packages
import streamlit as st

## Visu
def main():
    st.title("ESSAI")
    st.write('wow what a change')
    st.write('is this working in private ?')
    name = st.text_input("enter your name")
    if name :
        st.success(f"Hello {name}")
    
if __name__=="__main__":
    main()
