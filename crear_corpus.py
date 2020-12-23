# Autores: Ana Maria Casado y Ana Sanmartin
#
# Este script de realizar el preprocesamiento de los textos tokenizando y limpiando el texto.

# Paquetes necesarios para el funcionamiento del programa
import spacy
import nltk
from nltk import SnowballStemmer
import os
import pandas as pd

def pre_procesar_texto(texto: str, mode: str):
    
    path = os.getcwd()
    stop_words = []
    
    #Eliminacion de los stopwords
    #nlp = spacy.load('es_core_news_sm') # Se decidió no utilizar las stopwords de spacy porque nos parecían incompletas
    f = open(path +"/stop_words.txt","r") #Usamos nuestras propias stopwords

    sw = f.readlines()

    # Strips the newline character 
    for line in sw: 
        stop_words.append(line.strip()) 
    
    #Tokenizacion y limpieza del texto
    
    doc = nlp(texto) # Crea un objeto de spacy tipo nlp
    tokens = [token for token in doc if not token.is_punct and len(token) > 2] # Convierte las palabras en tokens
    clean = [token for token in tokens if not str(token) in stop_words and not isinstance(token, int)] #Elimina las stopwords del texto
    
    if(mode == 'lema'):
        clean = [token.lemma_.lower() for token in clean]

    elif(mode == 'stem'): 
        pass
        spanishstemmer=SnowballStemmer('spanish')
        clean = [spanishstemmer.stem(token) for token in clean]
    return clean

        
