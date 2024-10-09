# import streamlit as st
# import numpy as np
# import pickle
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# #Load the LSTM Model
# model=load_model('next_word_lstm.h5')

# #3 Laod the tokenizer
# with open('tokenizer.pickle','rb') as handle:
#     tokenizer=pickle.load(handle)

# # Function to predict the next word
# def predict_next_word(model, tokenizer, text, max_sequence_len):
#     token_list = tokenizer.texts_to_sequences([text])[0]
#     if len(token_list) >= max_sequence_len:
#         token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
#     token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
#     predicted = model.predict(token_list, verbose=0)
#     predicted_word_index = np.argmax(predicted, axis=1)
#     for word, index in tokenizer.word_index.items():
#         if index == predicted_word_index:
#             return word
#     return None

# # streamlit app
# st.title("WordWhiz")
# input_text=st.text_input("Enter the sequence of Words","To be or not to")
# if st.button("Predict Next Word"):
#     max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
#     next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
#     st.write(f'Next word: {next_word}')
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM Model
model = load_model('next_word_lstm.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Streamlit app
st.markdown(
    """
    <style>
        body {
            background-color: #f0f0f5;
        }
        .title {
            text-align: center;
            font-size: 48px;  /* Increased font size */
            color: #4CAF50;
        }
        .stButton {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 5px;
            display: block;
            margin: 20px auto;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 class='title'>WordWhiz 🌟</h1>", unsafe_allow_html=True)

input_text = st.text_area("Enter the sequence of Words", "To be or not to", height=100)

if st.button("Predict Next Word"):
    with st.spinner("Predicting..."):
        max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)

        # Display the result
        if next_word:
            st.success(f'The predicted next word is: **{next_word}** 😊')
        else:
            st.error("Sorry, I couldn't predict the next word. 😞")
