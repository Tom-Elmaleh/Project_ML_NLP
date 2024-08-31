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
Second, we implemented different models of Summarization, Translation, Text Generation and Sentiment Analysis using transformers and Hugging face preconfigured models that uses the customer reviews data.
Then, we used embeddings and bigrams to identify frequent words on the text.
Finally, we set up an information retrieval model based on a query specified by the user.

### Streamlit Application

Link towards the first version of the streamlit app developed : **https://init-version-nlp.streamlit.app/** <br>
On the streamlit application, the following models have been implemented :
  - Sentiment Analysis (supervised model not trained on the dataset) <br>
  - Summarization (supervised model not trained on the dataset) <br>
  - Information retrieval (model trained on the dataset) <br>
  - QA (supervised model not trained on the dataset) <br>
  - Rating prediction (supervised model not trained on the dataset) <br>

We also created a second app accesible through this link : **https://tom-elmaleh-project-ml-nlp-app-v3-lduq3e.streamlit.app/** <br>
On this application a function (truncate_review) has been implemented to check if the length of the input respects the token limitations of the model and if it’s not the case, the input is truncated.

#### Remarks on the Streamlit application

When trying to execute different tasks on the applications, sometimes we have noticed that the application can have some crashes with an error. 
So in that case, the best way to solve this issue is to reboot the application in the streamlit.
As for the reason associated to this problem, we think that it could be related to the high amount of storage needed for the hugging face models as well as the function that is used to check if the input has to be truncated. 
We also think that the summary model could use too much resources.
We have tried to use a the decorator @st.cache on our streamlit application but still, we didn’t manage to correct this issue.
