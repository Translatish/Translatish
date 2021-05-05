import pandas as pd
import numpy as np
import os
import string
from string import digits
import re

import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def generate_batch(X , y ,max_length_src,max_length_tar, num_decoder_tokens,batch_size,all_hindi_words,all_eng_words,input_token_index,target_token_index):
    ''' Generate a batch of data '''
    while True:
        for j in range(0, len(X), batch_size):
            global list_of_not_found
            list_of_not_found=[]
            encoder_input_data = np.zeros((batch_size, max_length_src),dtype='float32')
            decoder_input_data = np.zeros((batch_size, max_length_tar),dtype='float32')
            decoder_target_data = np.zeros((batch_size, max_length_tar, num_decoder_tokens),dtype='float32')
            for i, (input_text, target_text) in enumerate(zip(X[j:j+batch_size], y[j:j+batch_size])):
                for t, word in enumerate(input_text.split()):
                    if word not in all_eng_words:
                    
                      list_of_not_found.append(word)
                      continue
                    else:
                      encoder_input_data[i, t] = input_token_index[word] # encoder input seq
                for t, word in enumerate(target_text.split()):
                    if word not in all_hindi_words:
                      continue
                    else:
                      if t<len(target_text.split())-1:
                          decoder_input_data[i, t] = target_token_index[word] # decoder input seq
                      if t>0:
                          decoder_target_data[i, t - 1, target_token_index[word]] = 1.
            yield([encoder_input_data, decoder_input_data], decoder_target_data)

def decode_sequence(input_seq,encoder_model,decoder_model,target_token_index,reverse_target_char_index):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index['START_']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += ' '+sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '_END' or
           len(decoded_sentence) > 50):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence

def createModel(eng_sen):
    print(eng_sen)
    #return (os.getcwd())
    
    data = pd.read_csv('./app1/preprocessed_csv.csv')
    all_eng_words=set()

    for eng in data['english_sentence']:
        for word in str(eng).split(" "):
            if word not in all_eng_words:
                all_eng_words.add(word)

    all_hindi_words=set()
    for hin in data['hindi_sentence']:
        for word in hin.split():
            if word not in all_hindi_words:
                all_hindi_words.add(word)

    input_words = sorted(list(all_eng_words))
    target_words = sorted(list(all_hindi_words))
    num_encoder_tokens = len(all_eng_words)
    num_decoder_tokens = len(all_hindi_words)
    
    num_decoder_tokens += 1
    num_encoder_tokens +=1
    
    input_token_index = dict([(word, i+1) for i, word in enumerate(input_words)])
    target_token_index = dict([(word, i+1) for i, word in enumerate(target_words)])

    reverse_input_char_index = dict((i, word) for word, i in input_token_index.items())
    reverse_target_char_index = dict((i, word) for word, i in target_token_index.items())

    data['length_eng_sentence']=data['english_sentence'].apply(lambda x:len(str(x).split(" ")))
    data['length_hin_sentence']=data['hindi_sentence'].apply(lambda x:len(x.split(" ")))

    max_length_src=max(data['length_hin_sentence'])
    max_length_tar=max(data['length_eng_sentence'])

    data=data[data['length_eng_sentence']<=20]
    data=data[data['length_hin_sentence']<=20]  
    

    encoder_inputs = Input(shape=(None,))
    latent_dim=256
    enc_emb =  Embedding(num_encoder_tokens,latent_dim, mask_zero = True)(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None,))
    dec_emb_layer = Embedding(num_decoder_tokens, latent_dim, mask_zero = True)
    dec_emb = dec_emb_layer(decoder_inputs)

    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb,
                                        initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    batch_size=128
    model.load_weights('./app1/nmt_weights.h5')   
    #generate_batch(data['english_sentence'],data['hindi_sentence'],max_length_src,max_length_tar, num_decoder_tokens,batch_size,all_hindi_words,all_eng_words,input_token_index,target_token_index) 

    encoder_model = Model(encoder_inputs, encoder_states)

    # Decoder setup
    # Below tensors will hold the states of the previous time step
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    dec_emb2= dec_emb_layer(decoder_inputs) # Get the embeddings of the decoder sequence

    # To predict the next word in the sequence, set the initial states to the states from the previous time step
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
    decoder_states2 = [state_h2, state_c2]
    decoder_outputs2 = decoder_dense(decoder_outputs2)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs2] + decoder_states2)

    batch_size=1
    x=pd.Series([eng_sen])
    y=pd.Series([' ']*len(x))
    k=0
    train_gen = generate_batch(x[k:],y[k:],max_length_src,max_length_tar, num_decoder_tokens,batch_size,all_hindi_words,all_eng_words,input_token_index,target_token_index) 
    
    
    (input_seq, actual_output), _ = next(train_gen)
    decoded_sentence = decode_sequence(input_seq,encoder_model,decoder_model,target_token_index,reverse_target_char_index)     
    
    return (decoded_sentence[:-4])


#print(createModel(['the cows are fairly good milkyielders']))

