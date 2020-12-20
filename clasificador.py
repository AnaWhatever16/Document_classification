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

temas = ["Deportes", "Politica", "Salud"]

def clasificador_documentos(directorio, n_min, rango, glosario, modelo):
    
    modelos_tema = {} 
    sim_tema = {}

    path = ""
    path_glosario = directorio + "/Glosario/"
    dictionary = create_dictionary(path_glosario)

    f = open(directorio +"/stop_words.txt","r")
    sw = f.readlines()

    # Strips the newline character 
    for line in sw: 
        STOP_WORDS.append(line.strip()) 

    for i in temas:
        path = directorio + "/Documentos/" + i + "/"
        
        train_docs = []
        for j in range(0, n_min-1):
            f = open(path + i.lower() + str(j+1) + ".txt","r")
            files = f.read()
           
            #Se almacenan todos los documentos en una lista para poder procesarlos conjuntamente
            train_docs += [files]

        #Pre-procesamiento de los documentos de train y creacion de los modelos   
        train_bow = process_text_doc2bow(train_docs, dictionary)  
        modelos_tema[i], sim_tema[i] = crear_modelos(train_bow, dictionary, modelo)

    #Documentos para test 
    res = open(resultados,"r+")
    res.truncate(0)
    res.close()

    for i in temas:
        path = directorio + "/Documentos/" + i + "/"
        for j in range(n_min-1, n_min + rango -1):
            doc_path = path + i.lower() + str(j+1) + ".txt"
            test_doc=""
            f = open(doc_path,"r")
            test_doc = f.read()
            f.close()
            values={}
            # Dependiendo del modelo a utilizar se llamará a las funciones X_model
            values, print_thing = lanzar_clasificador(test_doc, dictionary, modelos_tema, sim_tema)
            guardar_resultados(resultados, doc_path, values)

        print(print_thing)
    
def guardar_resultados(path_results, path_doc, values):
    # Guardar txt con los documentos y los 3 valores para cada glosario
    # Mostrar por pantalla
    # Por modelo, ppppprecisión (relevantes/recuperados) y exhaustividad (los que son/los relevantes)
    # Dibujitos

    res = open(path_results, "a")
    res.write(path_doc + "\t" + str(values)+"\n")
    res.close()
    pass

def tfidf_model(bow, dictionary):     
    tfidf = models.TfidfModel(bow)
    index = similarities.MatrixSimilarity(bow, num_features=len(dictionary))
    return tfidf, index
           
           
def word2vec_model(bow, glosario, dictionary, path_glosario):
    #w2v_vector_size = 100
       #model_w2v = models.Word2Vec(sentences=texto_limpio, window=5,
       #                     workers=12, vector_size=w2v_vector_size, min_count=1, seed=50)
       #model_w2v.save(directorio + "/Modelos/word2vec_" + i + ".model")
       #guardar_resultados()
    pass
    
    
#Llamada al modelo de naive bayes
def naivebayes_model(bow, glosario, dictionary, path_glosario):
    #guardar_resultados()
    pass


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


#Realización del pre-proceso de los textos
def process_text_doc2bow(docs, dictionary):
    corpus = clean_docs(docs)
    bow = create_bow_from_dict(corpus, dictionary)
    return bow


#Se convierte los textos en un diccionario y se almacena en un documentos llamado corpus
def create_dictionary(path_glosario, pathname= None):
    dictionary = corpora.Dictionary()
    
    for filename in os.listdir(path_glosario):
        f2 = open(path_glosario + filename, "r")
        glosario = f2.read()
        #glosario_clean =  wordpunct_tokenize(glosario) ## GUARDAR LOS TRES DICCIONARIOS JUNTOS!!!
        tokens = [word for word in glosario.split()]
        dictionary.add_documents([tokens])

    if pathname:
        dictionary.save(pathname+"/corpus.dict")
    return dictionary
    
    
def create_bow_from_dict(corpus, dictionary, pathname=None):
    bow = [dictionary.doc2bow(text) for text in corpus]
    if pathname:
        corpora.MmCorpus.serialize(pathname+'/vsm_docs.mm', bow)
    return bow


# #La función launch_glosario hará la comparación entre el glosario y los documentos
# def launch_glosario_tfidf(glosario, tfidf, bow, dictionary):
#     glosario = tfidf[dictionary.doc2bow(glosario)]
    
#     return enumerate(index[glosario])# key=itemgetter(1), reverse=True)
    
    
#Dependiendo del valor de la variable modelo, la función crear_modelos utilizará la llamada al proceso correspondiente    
def crear_modelos(train_bow, dictionary, m):
    #if(m == 0):
    model, index = tfidf_model(train_bow, dictionary)
        #Crear diccionario con glosario
    # if(m == 1):
    #     word2vec_model(train_bow, glosario, dictionary, path_glosario)
    # if(m == 2):
    #     naivebayes_model(train_bow, glosario, dictionary, path_glosario)
    
    return model, index

def lanzar_clasificador(test_doc, dictionary, modelos_tema, sim_tema):
    # for doc in test_docs:
        #Pre-procesamiento del documento de test    
    test_bow = process_text_doc2bow([test_doc], dictionary) 
    values = {}
    for i in temas:
        tfidf = modelos_tema[i]
        index = sim_tema[i]
        doc_tfidf = tfidf[test_bow[0]]
        values[i] = index[doc_tfidf]

    return values, index[doc_tfidf]
   
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
    print("ERROR: Porfavor introduzca palabras válidas para directorio")
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
# else:
#     print("ERROR: Porfavor introduzca palabras válidas")
#     exit()
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



