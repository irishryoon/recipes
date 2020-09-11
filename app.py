import streamlit as st
import h5py
import numpy as np
import pickle 
from tensorflow.keras.models import load_model 

class MetricsAtK:
    def __init__(self, k):
        self.k = k

    def _prediction_tensor(self, y_pred):
        # Given tensor y_pred of floats (predicted probabilities), 
        # return a tensor of same shape with 1 in locations with "k" top probabilities. 
        y_pred = tf.convert_to_tensor(y_pred)
        topk_indices = tf.nn.top_k(y_pred, k = self.k, sorted = True).indices
        ii, _ = tf.meshgrid(tf.range(tf.shape(y_pred)[0]), tf.range(6), indexing='ij')
        index_tensor = tf.reshape(tf.stack([ii, topk_indices], axis=-1), shape=(-1, 2))
        index_tensor = tf.cast(index_tensor, tf.int64)
        n_nzero = tf.shape(index_tensor)[0]

        sparse = tf.SparseTensor(indices = index_tensor, values=tf.ones(n_nzero), dense_shape=tf.shape(y_pred, out_type=tf.dtypes.int64))
        sparse = tf.sparse.reorder(sparse)
        pred = tf.sparse.to_dense(sparse)
        pred = tf.cast(pred, tf.float32)

        return pred

    def _TP_at_k(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, dtype = tf.float32)
        # compute true positives for each sample
        pred = self._prediction_tensor(y_pred)

        TP = K.sum(tf.multiply(pred, y_true), axis = 1)
        TP = tf.cast(TP, tf.float32)
        
        return TP
    
    def precision_at_k(self, y_true, y_pred):
        TP = self._TP_at_k(y_true, y_pred)
        precisions = TP / self.k
        
        # compute median
        mid = tf.shape(precisions)[0]//2 + 1
        precision_median = tf.nn.top_k(precisions, mid).values[-1]
        return precision_median
    
    def recall_at_k(self, y_true, y_pred):
        TP = self._TP_at_k(y_true, y_pred)
        y_true = tf.convert_to_tensor(y_true, dtype = tf.float32)
        recalls = TP / K.sum(y_true, axis = 1)
        
        # compute median
        mid = tf.shape(recalls)[0]//2 + 1
        recall_median = tf.nn.top_k(recalls, mid).values[-1]
        return recall_median
    
    def F1_at_k(self, y_true, y_pred):
        TP = self._TP_at_k(y_true, y_pred)
        y_true = tf.convert_to_tensor(y_true, dtype = tf.float32)
        precisions = TP / self.k
        recalls = TP / K.sum(y_true, axis = 1)
        f1 = (2 * precisions * recalls) / (precisions + recalls + K.epsilon())
        
        # compute median
        mid = tf.shape(f1)[0]//2 + 1
        f1_median = tf.nn.top_k(f1, mid).values[-1]
        return f1_median

def main():

    st.title("Cooking Ingredients Recommender")
    
    # try options with all ingredients 
    # maybe only keep ingredients that occur frequently enough? 

    # load tokenizer
    with open('models/tokenizer.pickle', 'rb') as handle:
        t = pickle.load(handle)
    index_to_words=dict(map(reversed,t.word_index.items()))
    n_words = len(t.word_index) 
    
    # Because of "multiselect" limit, only present frequently occurring ingredients
    freq_ing = [key for (key, value) in t.word_counts.items() if value > 40]
    freq_ing.sort()
    ing_list = st.multiselect('Select ingredients',freq_ing)


    # load trained model
    metrics = MetricsAtK(k=6)
    model = load_model('models/multilabel_NN/model.h5', custom_objects= {'precision_at_k':metrics.precision_at_k,
                                                                                  'recall_at_k':metrics.recall_at_k,
                                                                                  'F1_at_k': metrics.F1_at_k})

    # get input ingredients 
    #ing = input("Enter ingredients: ") # ingredients separated by space
    #ing_list = ing.split()
    # check that all ingredients exist in the tokenizer

    # prepare array 
    ing_idx = [[t.word_index[item]-1 for item in ing_list]] # adjust indexing to start at 0
    X = np.array(ing_idx)
    
    # make predictions
    y = model.predict(X)
    rec_idx = y[0,:].argsort()[::-1][:6].tolist()
    rec_str = [index_to_words[item+1] for item in rec_idx] 
    print(rec_str)


if __name__ == '__main__':
    main()