from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.core.files.storage import default_storage
import moviepy.editor as mp
import speech_recognition as sr
from rest_framework import status
from pydub import AudioSegment
from . import test

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model




def videoToEnglish(file_name):
    clip = mp.VideoFileClip(file_name)
    clip.audio.write_audiofile(r"audio.mp3")
    sound = AudioSegment.from_mp3("audio.mp3")
    sound.export("transcript.wav", format="wav")

    r = sr.Recognizer()
    AUDIO_FILE = "transcript.wav"
    english=""
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)  # read the entire audio file                  
        english = r.recognize_google(audio)
        print("Transcription: " + english)
    return english


@api_view(['GET', 'POST'])
def home(request):
    #try:
    poste_file=request.FILES.getlist('myFile')[0]
    file_name = default_storage.save(poste_file.name, poste_file)
    
    print(file_name)
    english=videoToEnglish(file_name)
    
    default_storage.delete(file_name)
    hindi=test.createModel(english)
    default_storage.delete("audio.mp3")
    default_storage.delete("transcript.wave")
    print(hindi)
    return Response({
        'hindi':hindi,
        'english':english
        })
    # except  Exception as e:
    #     return Response({"message":"Something went wrong!!"+str(e)},status= status.HTTP_400_BAD_REQUEST)
    