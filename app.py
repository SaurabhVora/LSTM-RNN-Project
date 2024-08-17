import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#! Load the trained model and tokenizer
model = load_model('hamlet.keras')
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

#! Function to predict the next word in a sequence
def predict_next_words(model, tokenizer, text, max_sequence_len):
    #* Convert the input text to a sequence of tokens
    token_list = tokenizer.texts_to_sequences([text])[0]

    #* Trim the token list to the appropriate length
    token_list = token_list[-(max_sequence_len-1):] if len(token_list) >= max_sequence_len else token_list

    #* Pad the sequence to the required length
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

    #* Predict the next word
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]

    #* Reverse the tokenizer word index to map indices back to words
    reverse_word_index = {index: word for word, index in tokenizer.word_index.items()}

    #* Return the predicted word or None if not found
    return reverse_word_index.get(predicted_word_index, None)


#! Streamlit UI
st.title("Next Word Prediction with LSTM")

st.markdown("""
    This app predicts the next word in a given sentence using an LSTM model trained on the text of Hamlet.
    Simply enter a partial sentence, and the model will suggest the next word.
""")

input_text = st.text_input('Enter a Sentence or Phrase')

if st.button('Predict Next Word'):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_words(model, tokenizer, input_text, max_sequence_len)
    if next_word:
        st.success(f"**Next Word Prediction:** {next_word}",icon="🤖")
    else:
        st.error("Could not predict the next word. Please try a different input.")

st.markdown("### Example Sentences to Try:")
st.write("- To be or not to")
st.write("- The prince of Denmark")
st.write("- What a piece of work is man")

st.sidebar.header("About")
st.sidebar.info("""
    This app is powered by a Keras LSTM model trained on Shakespeare's Hamlet.
    The model predicts the next word in a sequence based on the input text.
""")
