import streamlit as st 
import pandas as pd
from transformers import pipeline
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from transformers import AutoTokenizer
import nltk
nltk.download('stopwords')
import torch 
  
# Preprocessing for sentiment analysis
@st.cache(allow_output_mutation=True)
def clean_sentence(val):
    """ Cette fonction permet de supprimer les caractères spéciaux et les mots de taille inférieure à 2
    pour la tâche de summarization"""
    # Remplacement des valeurs \n
    val = val.replace('\n','')
    # Suppression des caratères spéciaux
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub('', val).lower()
    sentence = sentence.split(" ")
    for word in list(sentence):
        if len(word) <= 2:
            sentence.remove(word)
    sentence = " ".join(sentence) 
    return sentence

@st.cache(allow_output_mutation=True)
def preprocess(text):
    # Remove special characters and put in lower case
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    # Tokenization, remove stop words, and stemming
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer('english')
    tokens = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# Token limit
@st.cache(allow_output_mutation=True)
def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

@st.cache(allow_output_mutation=True)
def truncate_review(review, max_length, model_name):
    """Function to take into account the token limit of models by truncating tokens"""
    # Load the tokenizer
    tokenizer = load_tokenizer(model_name)
    # Tokenize the review and truncate tokens
    tokens = tokenizer(review, return_tensors="pt", truncation=True)
    tokens_truncated = {key: value[:, :max_length] for key, value in tokens.items()}
    # Convert the truncated tokens to a string
    review_truncated = tokenizer.decode(tokens_truncated["input_ids"][0], skip_special_tokens=True)
    return review_truncated

# Summarizer
@st.cache(allow_output_mutation=True)
def load_summarizer():
    return pipeline("summarization", model='sshleifer/distilbart-cnn-12-6')

@st.cache(allow_output_mutation=True)
def apply_summarization(text):
    summarizer = load_summarizer()
    text_truncated = truncate_review(text, max_length=1024, model_name='sshleifer/distilbart-cnn-12-6')
    summary = summarizer(text_truncated, max_length=100, min_length=20, do_sample=False)[0]["summary_text"]
    summary = summary.strip()
    return summary

@st.cache(allow_output_mutation=True)
def load_classifier():
    return  pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")

# Sentiment analysis model
@st.cache(allow_output_mutation=True)
def apply_sentiment_analysis(text):
    """Function which applies a sentiment analysis"""
    text = clean_sentence(text)
    classifier = load_classifier()
    text = truncate_review(text, max_length=128, model_name='distilbert-base-uncased-finetuned-sst-2-english')
    sentiments = classifier(text)[0]
    return sentiments['label'], sentiments['score']


# QA model
@st.cache(allow_output_mutation=True)
def load_qa():
    return  pipeline(model="deepset/roberta-base-squad2")

@st.cache(allow_output_mutation=True)
def apply_qa(text,query):
    """Function to apply a QA model"""
    oracle = load_qa()
    text = truncate_review(text, max_length=386, model_name='deepset/roberta-base-squad2')
    query = truncate_review(query, max_length=64, model_name='deepset/roberta-base-squad2')
    answer = oracle(question=query, context=text)['answer']
    return answer

@st.cache(allow_output_mutation=True)
def load_text_classif():
    return  pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Rating prediction model
@st.cache(allow_output_mutation=True)
def apply_prediction(review):
    """Function to apply a the rating prediction model"""
    sentiment_pipeline = load_text_classif()
    review = truncate_review(review, max_length=512, model_name='nlptown/bert-base-multilingual-uncased-sentiment')
    result = sentiment_pipeline(review)[0]
    return result['label'],result['score']

@st.cache(allow_output_mutation=True)
# Information retrieval using TF-IDF and cosine similarity
def retrieve_information(query):
    """IR system using TF-IDF and cosine similarity to give the best results associated with the query"""
    df = pd.read_csv('train_set.csv',index_col=0)
    # Combine relevant informations in one column
    df['information'] =  df['avis_en'] + df['produit'] + df["assureur"].str.lower()   
    # Application preprocessing
    df['information'] = df['information'].apply(preprocess)
    # Use Tf_idf vectorizer for the combined_text
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['information'])
    # Vectorize the query
    query_vectorized = vectorizer.transform([query])
    # Compute cosine similarity between the query and the tfidf_matrix
    similarity_scores = cosine_similarity(query_vectorized, tfidf_matrix).flatten()
    # Add similarity scores to df and sort by similarity 
    df['similarity'] = similarity_scores
    df = df.sort_values(by='similarity', ascending=False)
    return df[['assureur', 'produit','avis_en','note','similarity']].head(10)

    
st.title("Machine Learning for NLP Project")
st.subheader("For each task, follow the instructions written and then click on the button *Analyze* to run the models")
selected_action = st.selectbox("***Select the action that you want to perform***", ["Sentiment Analysis", "Summary","Information Retrieval","QA","Rating Prediction"])

if selected_action == "Summary":
    st.write("****Summarization is performed with a supervised model****")
    text_input_summary = st.text_area("***Enter the text that you want to summarize***", height=100)
    analyze_button_summary = st.button("****Analyze****", key="analyze_button_summary")
    if text_input_summary and analyze_button_summary:
        with st.spinner("**Summarization ongoing...**"):
            summary = apply_summarization(text_input_summary)
            st.write(f"***{summary}***")

elif selected_action == "Sentiment Analysis":
    st.write("****Sentiment Analysis is performed with a supervised model****")
    text_input_sa = st.text_area("***Enter your text for sentiment analysis***", height=5)
    analyze_button_sa = st.button("****Analyze****", key="analyze_button_prediction")
    if text_input_sa and analyze_button_sa:
        with st.spinner("**Sentiment Analysis ongoing...**"):
            sentiment_label, sentiment_score = apply_sentiment_analysis(text_input_sa)
            st.write(f"***Sentiment Detected : {sentiment_label}***")
            st.write(f"***Score : {sentiment_score}***")

elif selected_action == "Information Retrieval":
    st.write("****The Information Retrieval system is based on the insurance dataset. It uses the reviews of users and its based on cosine similarity.****")
    query = st.text_input("***Write the query that will be associated to the information retrival system.*** *Example : best auto insurance*")
    analyze_button_ir = st.button("****Analyze****", key="analyze_button_ir")
    if query and analyze_button_ir:
        with st.spinner("**Please wait for the result...**"):
            st.write("****Most similar elements associated to your query :****")
            st.dataframe(retrieve_information(query))
 
elif selected_action == "QA":
    st.write("****Question Answering is performed with a supervised model****")
    text_input_qa = st.text_input("***Enter the text that you want to query***")
    query_qa = st.text_input("***Write your query***")
    analyze_button_qa = st.button("****Analyze****", key="analyze_button_qa")
    if text_input_qa and analyze_button_qa and query_qa :
        with st.spinner("**Please wait for the answer...**"):
            answer = apply_qa(text_input_qa,query_qa)
            st.write(f"***{answer}***")

elif selected_action == "Rating Prediction":
    st.write("****Rating Prediction is performed with a supervised model****")
    review = st.text_area("***Enter your review***", height=5)
    analyze_button_rp = st.button("****Analyze****", key="analyze_button_rp")
    if review and analyze_button_rp :
        with st.spinner("**Rating prediction of the review ongoing...**"):
            rating_prediction, score_prediction = apply_prediction(review)
            st.write(f"***Rating predicted : {rating_prediction}***")
            st.write(f"***Score : {score_prediction}***")