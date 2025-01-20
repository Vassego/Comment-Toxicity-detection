import streamlit as st
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import TextVectorization  
from keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding
from keras.models import load_model



def predict(input_text):
    
    print(os.path.join('data', 'train.csv'))
    df = pd.read_csv(os.path.join('data', 'train.csv'))
    X = df['comment_text'] 
    y = df[df.columns[1]].values
    MAX_FEATURES = 200000    
    vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=1800, output_mode='int')
    vectorizer.adapt(X.values)
    
    
    new_model = load_model(os.path.join('models', 'Toxicity_final.h5'))
    comment = input_text
    vectorized_comment = vectorizer(comment)
    
    res = (new_model.predict(np.expand_dims(vectorized_comment, 0)) > 0.5).astype(int)
    if res == 0:
        return "The comment is Not Toxic"
    else:
        return "The comment is Toxic"
    

# Streamlit
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #2c003e, #6a0dad); /* Dark Purple gradient */
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #ffffff;
        font-family: 'Verdana', sans-serif;
    }

    input[type="text"]::placeholder {
        color: black !important; 
        opacity: 1; 
    }

    input[type="text"] {
        background-color: rgba(255, 255, 255, 0.9);
        color: #333;
        border: 2px solid #8e44ad;
        border-radius: 10px;
        padding: 12px;
        font-size: 16px;
        width: 100%;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .stButton>button {
        background-color: #5a189a; /* Vibrant purple */
        color: white;
        font-size: 16px;
        font-weight: bold;
        border-radius: 12px;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #3c096c; 
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    .stMarkdown h3 {
        font-size: 2rem; /* Larger emoji size */
        line-height: 1.5;
        color:#ea0c4b;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
    }
    hr {
        border: 1px solid #8e44ad;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🔍 Comment Toxicity Detection Model")
st.markdown(
    """
    ### Welcome!
    Enter your comment below to find out whether it's **toxic** or **non-toxic**. 
    Our advanced AI model analyzes the comment for harmful language in just a second! 
    """
)

st.markdown("### 📝 **Type Your Comment**")  
user_input = st.text_input("", placeholder="Enter your comment here...")

col1, col2 = st.columns([1, 2])  
with col1:
    if st.button("Analyze Comment"):
        if user_input.strip():
            model_output = predict(user_input)
            st.success(f"**Prediction:** {model_output}")
        else:
            st.error("⚠️ Please enter a valid comment.")

st.markdown("---")
st.info(
    "🔔 **Tip:** This tool works best with complete sentences or phrases. "
)
