import h5py
import numpy as np
import pickle 
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model 
from tensorflow.keras import backend as K

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

@st.cache(show_spinner = False)
def load_cuisine():
    with open("cuisine.pickle", "rb") as f:
        cuisine = pickle.load(f)
    return cuisine

@st.cache(show_spinner = False)
def load_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
        t = pickle.load(handle)
    index_to_words=dict(map(reversed,t.word_index.items()))
    n_words = len(t.word_index) 
    
    # Because of "multiselect" limit, only present frequently occurring ingredients
    freq_ing = [key for (key, value) in t.word_counts.items() if value > 40]
    freq_ing.sort()
    return t.word_index, index_to_words, freq_ing

@st.cache(allow_output_mutation=True, show_spinner = False)
def load_trained_model():
    model = load_model('app_model.h5')
    model.summary() # THIS IS IMPORTANT to avoid caching error
    return model

def main():

    st.title("Cooking Ingredients Recommender")
    st.subheader("Select ingredients")

    cuisine = load_cuisine()
    word_index, index_to_words, freq_ing = load_tokenizer()
    model = load_trained_model()

    ing_list = st.multiselect('Type and select ingredients.',freq_ing)
    button = st.button("Click here to receive recommendations")

    # button indicating that user finished selecting ingredients
    if button:
        # prepare array 
        ing_idx = [[word_index[item]-1 for item in ing_list]] # adjust indexing to start at 0
        X = np.array(ing_idx)

        # make predictions
        y = model.predict(X)
        rec_idx = y[0,:].argsort()[::-1][:15].tolist()
        rec_str = [index_to_words[item+1] for item in rec_idx] 

        # remove water
        if "water" in rec_str:
            rec_str.remove("water")

        # remove cuisine type
        for item in cuisine:
            if item in rec_str:
                rec_str.remove(item)

        rec_str = rec_str[:6]
        st.subheader("Recommended ingredients")
        for item in rec_str:
            item = item.replace("_", " ")
            st.write(item)

if __name__ == '__main__':
    main()