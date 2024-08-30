ESILV - Machine Learning for NLP - Project 2024


<br>
<h1 align="center"> Machine Learning for NLP </h1>


### Introduction
This repository contains all files related to the final project of the course Machine Learning for NLP.  


### Purpose
This project aims to use show the use of *Natural Language Processing*  using different techniques in order to analyze customer reviews for insurance.


### Implementation

In order to make this project, we followed several steps.
First, we started by exploring and cleaning the dataset.
Second, we implemented different models of Summarization, Translation, Text Generation and Sentiment Analysis using transformers and Hugging face preconfigured models that uses the customer reviews data
Then, we used embeddings and bigrams to identify frequent words on the text.
Finally, we set up an information retrieval model based on a query specified by the user.

### Streamlit Application

Link towards the app : **https://init-version-nlp.streamlit.app/** (Version 1)
On the streamlit application, the following models have been implemented :
▪ Sentiment Analysis (supervised model not trained on the dataset)
▪ Summarization (supervised model not trained on the dataset)
▪ Information retrieval (model trained on the dataset)
▪ QA (supervised model not trained on the dataset)
▪ Rating prediction (supervised model not trained on the dataset)

We also created a second app accessible here : **https://tom-elmaleh-project-ml-nlp-app-v3-lduq3e.streamlit.app/**
On this application a function ( truncate_review) has been implemented to check if the length of the input respects the token limitations of the model and if it’s not the case, the input is truncated.
