import streamlit as st 
from transformers import pipeline
import pandas as pd

st.write("Machine Learning for NLP Project")

selected_action = st.selectbox("Select the action that you want to perform", ["Prediction","Summary", "Explanation","Information Retrieval","RAG","QA"])

# Zone de texte en entrée
text_input = st.text_input("Enter the text", " ")

# Summary
if (selected_action == "Summary") & (text_input!=''):
    summarizer = pipeline("summarization")
    results = summarizer(text_input,max_length=30, min_length=10, do_sample=False)
    st.write("Résultats du modèle :")
    st.write(results)



# Prediction (2 points)
if (selected_action == "Prediction") & (text_input!=''):
    st.write("Prediction")
    # summarizer = pipeline("summarization")
    # results = summarizer(text_input,max_length=30, min_length=10, do_sample=False)
    # st.write("Résultats du modèle :")
    # st.write(results)


# Explanation 
if (selected_action == "Explanation") & (text_input!=''):
    st.write("Explanation")
    # summarizer = pipeline("summarization")
    # results = summarizer(text_input,max_length=30, min_length=10, do_sample=False)
    # st.write("Résultats du modèle :")
    # st.write(results)

# Information Retrieval  
if (selected_action == "Information Retrieval") & (text_input!=''):
    st.write("Information Retrieval ")
    # summarizer = pipeline("summarization")
    # results = summarizer(text_input,max_length=30, min_length=10, do_sample=False)
    # st.write("Résultats du modèle :")
    # st.write(results)


# RAG
if (selected_action == "RAG") & (text_input!=''):
    st.write("RAG")
    # summarizer = pipeline("summarization")
    # results = summarizer(text_input,max_length=30, min_length=10, do_sample=False)
    # st.write("Résultats du modèle :")
    # st.write(results)

# QA 
if (selected_action == "QA") & (text_input!=''):
    st.write("QA")
    oracle = pipeline(model="deepset/roberta-base-squad2")
    results = oracle(question="What is the problem of the customer?", context=text_input)['answer']
    # results = summarizer(text_input,max_length=30, min_length=10, do_sample=False)
    # st.write("Résultats du modèle :")
    # st.write(results)
