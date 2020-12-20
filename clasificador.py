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
modelo = 0

resultados = ""
STOP_WORDS = []

def clasificador_documentos(directorio, n_min, rango, glosario, modelo):
    
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
       path_glosario = directorio + "/Glosario/"
       for j in range(n_min, n_min + rango -1):
           #print(path + i.lower() + str(j+1) + ".txt")
           f = open(path + i.lower() + str(j+1) + ".txt","r")
           files = f.read()
           
           #Se almacenan todos los documentos en una lista para poder procesarlos conjuntamente
           doc += [files]
           
       #Pre-procesamiento de los documentos de test    
       bow, dictionary = process_text(doc)    
       
       # Dependiendo del modelo a utilizar se llamará a las funciones X_model
       lanzar_clasificador(bow, dictionary, path_glosario, modelo)


################################
# ALMACENAMIENTO DE RESULTADOS #
################################

def guardar_resultados(path_results):
    # Guardar txt con los documentos y los 3 valores para cada glosario
    # Mostrar por pantalla
    # Por modelo, ppppprecisión (relevantes/recuperados) y exhaustividad (los que son/los relevantes)
    # Dibujitos
    pass

#####################################
# MODELOS A UTILIZAR EN EL PROYECTO #
#####################################

#Dependiendo del valor de la variable modelo, la función lanzar_clasificador utilizará la llamada al proceso correspondiente    
def lanzar_clasificador(bow, dictionary, path_glosario, m):
    if(m == 0):
        tfidf_model(bow, dictionary, path_glosario)
    if(m == 1):
        word2vec_model(bow, dictionary, path_glosario)
    if(m == 2):
        naivebayes_model(bow, dictionary, path_glosario)
        
def tfidf_model(bow, dictionary, path_glosario):
       for filename in os.listdir(path_glosario):
           f2 = open(path_glosario + filename, "r")
           glosario = f2.read()
           glosario = wordpunct_tokenize(glosario)
       
       tfidf = models.TfidfModel(bow)
       sims = launch_glosario_tfidf(glosario, tfidf, bow, dictionary)

       #guardar_resultados()
           
           
def word2vec_model(bow, dictionary, path_glosario):
    #w2v_vector_size = 100
       #model_w2v = models.Word2Vec(sentences=texto_limpio, window=5,
       #                     workers=12, vector_size=w2v_vector_size, min_count=1, seed=50)
       #model_w2v.save(directorio + "/Modelos/word2vec_" + i + ".model")
       #guardar_resultados()
    pass
    
    
#Llamada al modelo de naive bayes
def naivebayes_model(bow, dictionary, path_glosario):
    #guardar_resultados()
    pass


###################################
# MÉTODOS PARA PREPROCESAR TEXTOS #
###################################

#Limpieza de los textos, se aplica un stemmer, se tokenizan las palabras, se eliminan las palabras de parada (stop_words)
# del documento creado a mano, si la palabra no tiene mayor longitud que 2, tambien se considera no relevante.
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

#Se convierte los textos en un diccionario y se almacena en un documentos llamado corpus
def create_dictionary(corpus, pathname=None):
    dictionary = corpora.Dictionary(corpus)
    if pathname:
        dictionary.save(pathname+"/corpus.dict")
    return dictionary
    
    
def create_bow_from_corpus(corpus, dictionary, pathname=None):
    bow = [dictionary.doc2bow(text) for text in corpus]
    if pathname:
        corpora.MmCorpus.serialize(pathname+'/vsm_docs.mm', bow)
    return bow
    
#Realización del pre-proceso de los textos
def process_text(docs):
    corpus = clean_docs(docs)
    dictionary = create_dictionary(corpus)
    bow = create_bow_from_corpus(corpus, dictionary)
    return bow, dictionary



#####################################
# COMPARACIÓN GLOSARIO - DOCUMENTOS #
#####################################

#La función launch_glosario hará la comparación entre el glosario y los documentos
def launch_glosario_tfidf(glosario, tfidf, bow, dictionary):
    glosario = tfidf[dictionary.doc2bow(glosario)]
    index = similarities.SparseMatrixSimilarity(
        bow, num_features=len(dictionary))
    return enumerate(index[glosario])# key=itemgetter(1), reverse=True)
    


   
###########################################
# MÉTODOS PARA OBTENER DATOS POR PANTALLA #
###########################################

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
                    
parser.add_argument('-m',
                    "--modelo",
                    type=int,
                    help="Modelo a utilizar para el clasificador. 0 = VSM con tf-idf, 1 = VSM (word2vec), 2 = Naive Bayes")

parser.add_argument('-v',
                    "--valores",
                    type=str,
                    help = "Path al documento donde guardar los los resultados")


                    

# Parseo de los argumentos
arguments = vars(parser.parse_args())

if arguments['directorio']:
    directorio = arguments['directorio']
else:
    print("ERROR: Porfavor introduzca palabras válidas para el directorio")
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

if arguments['modelo']:
    if arguments['modelo'] > 0 and arguments['modelo'] < 2:
        modelo = arguments['modelo']
    else:
        print("ERROR: Introduzca un valor válido mayor que 0 y menor que 2 para un modelo válido")
        exit()
if arguments['valores']:
    resultados = arguments['valores']
else:
    print("ERROR: Porfavor introduzca palabras válidas para resultados")
    exit()

clasificador_documentos(directorio, nmin, rango, glosario, modelo)



