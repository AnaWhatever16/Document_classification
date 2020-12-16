import spacy
import re
import nltk
from nltk import SnowballStemmer

def procesar_texto(texto: str, mode: str, x: str):

    #Tokenización y limpieza del texto

    doc = nlp(texto) # Crea un objeto de spacy tipo nlp
    tokens = [t.orth_ for t in doc if not t.is_punct | t.is_stop] # Crea una lista con las palabras del texto y elimina los simbolos de puntuación

    #Normalización
    tokens = [t.lower() for t in words if t.isalpha()]
    
    #Aquí es dónde escogemos las palabras referentes. Se analizarán dos métodos diferentes, la proporción de aparición de una palabra y la tf-idf.
    if(x == 'frecuencia'):
        
    if(x == 'tfidf'):
        
    if(mode == 'lema'):
        #Lemalización
        lemmas = [tok.lemma_.lower() for tok in doc]
    
    elif(mode == 'stem'):
        #Stemming
        spanishstemmer=SnowballStemmer(‘spanish’)
        tokens = [t.lower() for t in words if t.isalpha()]
        stems = [spanishstemmer.stem(token) for token in tokens]
