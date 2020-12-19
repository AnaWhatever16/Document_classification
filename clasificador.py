# Autores: Ana Casado y Ana Sanmartin
#
# Este script recoge datos de la línea de comandos introducido por el usuario.

#Dependencias son:
#
#

# IMPORTS 
import os
import argparse
import pandas as pd
from operator import itemgetter
from crear_corpus import pre_procesar_texto
from gensim import corpora, models, similarities
from gensim.corpora.mmcorpus import MmCorpus
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import PorterStemmer

## ESTABLECIDOS POR EL USUARIO ##
directorio =""
nmin = 16
rango = 15
glosario = ""
STOP_WORDS = []

def clasificador_documentos(directorio, n_min, rango, glosario):
    
    temas = ["Deportes", "Politica", "Salud"]
    path = ""
    f = open(directorio +"/stop_words.txt","r")
    sw = f.readlines()

    # Strips the newline character 
    for line in sw: 
        STOP_WORDS.append(line.strip()) 

    
    # En esta función lo que se realizará es cargar los documentos a analizar en una lista
    
    for i in temas:
       doc = []
       path = directorio + "/Documentos/" + i + "/"
       path_query = directorio + "/Glosario/"
       for j in range(n_min, n_min + rango -1):
           #print(path + i.lower() + str(j+1) + ".txt")
           f = open(path + i.lower() + str(j+1) + ".txt","r")
           files = f.read()
           
           #texto_limpio = pre_procesar_texto(files.lower(), 'lema')
           doc += [files]
           
       bow, dictionary = process_text(doc)    
       #This is the ifidf model
       for filename in os.listdir(path_query):
           f2 = open(path_query + filename, "r")
           query = f2.read()
           query = tokens = wordpunct_tokenize(query)
           tfidf_model(bow, query, dictionary, filename)


       #w2v_vector_size = 100
       #model_w2v = models.Word2Vec(sentences=texto_limpio, window=5,
       #                     workers=12, vector_size=w2v_vector_size, min_count=1, seed=50)
       #model_w2v.save(directorio + "/Modelos/word2vec_" + i + ".model")
    
    # Albergar en un documento los textos correspondientes a cada glosario.
    

def tfidf_model(bow, query, dictionary, glosario):
       tfidf = models.TfidfModel(bow)
       sims = launch_query(query, tfidf, bow, dictionary)
       print("Para el tema: " + glosario)
       print("La relevancia por documento es: ")
       for doc, score in sims:
           print("Relevancia: ")
           print(score)
           print("Documento: ")

def clean_docs(docs):
    stemmer = PorterStemmer()
    final = []
    for doc in docs:
        tokens = wordpunct_tokenize(doc)
        clean = [stemmer.stem(token)
                 for token in tokens
                 if token.lower() not in STOP_WORDS
                 and len(token) > 2
                 and all(c.isalnum() for c in token)
                 ]
        final.append([stemmer.stem(word) for word in clean])
    return final

def process_text(docs):
    corpus = clean_docs(docs)
    dictionary = process_corpus(corpus)
    bow = create_bow_from_corpus(corpus, dictionary)
    return bow, dictionary

def process_corpus(corpus, pathname=None):
    dictionary = corpora.Dictionary(corpus)
    if pathname:
        dictionary.save(pathname+"/corpus.dict")
    return dictionary
    
def create_bow_from_corpus(corpus, dictionary, pathname=None):
    bow = [dictionary.doc2bow(text) for text in corpus]
    if pathname:
        corpora.MmCorpus.serialize(pathname+'/vsm_docs.mm', bow)
    return bow

def launch_query(query, tfidf, bow, dictionary):
    query = tfidf[dictionary.doc2bow(query)]
    index = similarities.SparseMatrixSimilarity(
        bow, num_features=len(dictionary))
    return enumerate(index[query])# key=itemgetter(1), reverse=True)
    
def get_clasificador(c):
   switcher = {
       0 : 'tfidf',
       1 : 'vsm'}
       
   return switcher.get(c)
   
######################
# PROGRAMA PRINCIPAL #
######################

# Argparse - Parámetros a ser introducidos por el usuario
parser = argparse.ArgumentParser(description="Search by terms")

parser.add_argument('-d',
                    "--directorio",
                    type=str,
                    help="Directorio general donde se encuentran las carpetas con los textos a procesar. El path debe de ser desde la raíz hasta la carpeta Documentos. Ej: .../Document_classification/Documentos")
                    
parser.add_argument('-n',
                    "--nmin",
                    type=int,
                    help="Posición a partir de la cual se utilizarán los documentos (Por ejemplo, a partir del documento 16)")
                    
parser.add_argument('-r',
                    "--rango",
                    type=int,
                    help="Número de documentos totales a utilizar para la clasificación de test (Por ejemplo si deseamos 15 documentos, del 16 al 30)")
                    
parser.add_argument('-g',
                    "--glosario",
                    type=str,
                    help="Directorio donde se encuentran los tres glosarios a utilizar. El path debe de ser desde la raíz hasta la carpeta donde se encuentren. Ej: .../Document_classification/Pre-Glosario")

                    

# Parseo de los argumentos
arguments = vars(parser.parse_args())

if arguments['directorio']:
    directorio = arguments['directorio']
else:
    print("ERROR: Porfavor introduzca palabras válidas")
    exit()
if arguments['nmin']:
    if arguments['nmin'] > 0:
        nmin = arguments['nmin']
    else:
        print("ERROR: El valor de N debe ser mayor que 0")
        exit()
if arguments['rango']:
    if arguments['rango'] > 0:
        modo = arguments['rango']
    else:
        print("ERROR: Introduzca un valor válido mayor que 0")
        exit()
if arguments['glosario']:
    glosario = arguments['glosario']
else:
    print("ERROR: Porfavor introduzca palabras válidas")
    exit()

clasificador_documentos(directorio, nmin, rango, glosario)



