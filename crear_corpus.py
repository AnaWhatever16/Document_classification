import spacy
from collections import Counter
import nltk
from nltk import SnowballStemmer
import os
import pandas as pd

def pre_procesar_texto(texto: str, mode: str):
    
    path = os.getcwd()
    stop_words = []
    
    #Eliminación de los stopwords
    nlp = spacy.load('es_core_news_sm') # Carga del modelo
    f = open(path +"/stop_words.txt","r")

    sw = f.readlines()

    # Strips the newline character 
    for line in sw: 
        stop_words.append(line.strip()) 
    
    
    
    #Tokenización y limpieza del texto
    
    doc = nlp(texto) # Crea un objeto de spacy tipo nlp
    tokens = [token for token in doc if not token.is_punct and len(token) > 2]
    clean = [token for token in tokens if not str(token) in stop_words and not isinstance(token, int)]
    
    if(mode == 'lema'):
        clean = [token.lemma_.lower() for token in clean]

    elif(mode == 'stem'):
        pass
        #spanishstemmer=SnowballStemmer('spanish')
        #clean = [spanishstemmer.stem(token) for token in clean]
    return clean
    #Aquí es dónde escogemos las palabras referentes. Se analizarán dos métodos diferentes, la proporción de aparición de una palabra y la tf-idf.

        
