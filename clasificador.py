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
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

## ESTABLECIDOS POR EL USUARIO ##
directorio =""
nmin = 16
rango = 15
modelo = 0

STOP_WORDS = []

def clasificador_documentos(directorio, n_min, rango, modelo):
    
    temas = ["Deportes", "Politica", "Salud"]
    path = ""
    f = open(directorio +"/stop_words.txt","r")
    sw = f.readlines()

    # Strips the newline character 
    for line in sw: 
        STOP_WORDS.append(line.strip()) 

    # En esta función lo que se realizará es cargar los documentos a analizar en una lista
    
    path_results = directorio + "/Resultados/"
    path_glosario = directorio + "/Glosario/"
    
    doc = []
    doc_id = []
    for i in temas:
        path = directorio + "/Documentos/" + i + "/"
        for j in range(n_min, n_min + rango -1):
            f = open(path + i.lower() + str(j+1) + ".txt","r")
            files = f.read()
	    
           #Se almacenan todos los documentos en una lista para poder procesarlos conjuntamente
            doc += [files]
            doc_id += [i.lower() + str(j+1)]

    dictionary = create_dictionary(path_glosario)

    #Pre-procesamiento de los documentos de test    
    bow = process_text(doc, dictionary)    
    
    #Esto es para train de Naive Bayes, cambiarme porfavor
    if(modelo == 2):
        doc = []
        label = []
        for i in temas:
            path = directorio + "/Documentos/" + i + "/"
            for j in range(n_min - 1):
                f = open(path + i.lower() + str(j+1) + ".txt","r")
                files = f.read()
                #Se almacenan todos los documentos en una lista para poder procesarlos conjuntamente
                doc += [files]
                label += [i.lower()]

    	#Pre-procesamiento de los documentos de test    
        bow_train = process_text(doc, dictionary)
        for j,l in enumerate(label):
            if l == 'deportes':
                label[j] = 0
            elif l == 'politica':
                label[j] = 1
            elif l == 'salud':
                label[j] = 2
        naivebayes_model(bow_train, bow, label, dictionary, path_glosario)
    	
    else:
    # Dependiendo del modelo a utilizar se llamará a las funciones X_model
    	lanzar_clasificador(bow, doc_id, dictionary, path_glosario, path_results, modelo)


################################
# ALMACENAMIENTO DE RESULTADOS #
################################

def guardar_resultados(ranking, doc_id, path_results):
    # Guardar txt con los documentos y los 3 valores para cada glosario
    # Mostrar por pantalla
    # Por modelo, ppppprecisión (relevantes/recuperados) y exhaustividad (los que son/los relevantes)
    # Dibujitos

    res = open(path_results, "w")
    for doc, score in ranking:
        doc_string = doc_id[doc] + ' ' + str(score) + '\n'
        res.write(doc_string)
    res.close()

#####################################
# MODELOS A UTILIZAR EN EL PROYECTO #
#####################################

#Dependiendo del valor de la variable modelo, la función lanzar_clasificador utilizará la llamada al proceso correspondiente    
def lanzar_clasificador(bow, doc_id, dictionary, path_glosario, path_results, m):
    if(m == 0):
        tfidf_model(bow, doc_id, dictionary, path_glosario, path_results)
    elif(m == 1):
        word2vec_model(bow, dictionary, path_glosario, path_results)
    elif(m == 2):
        naivebayes_model(bow, dictionary, path_glosario)
        
        
def tfidf_model(bow, doc_id, dictionary, path_glosario, path_results):
        tfidf = models.TfidfModel(bow)
        index = similarities.SparseMatrixSimilarity(bow, num_features=len(dictionary))

        for filename in os.listdir(path_glosario):
            f2 = open(path_glosario + filename, "r")
            glosario = f2.read()
            clean_glosario = wordpunct_tokenize(glosario)
            tfidf_glosario = tfidf[dictionary.doc2bow(clean_glosario)]
            sims = enumerate(index[tfidf_glosario])

            path_res_glosario = path_results+ "/tfidf/tfidf_" + filename
            guardar_resultados(sims, doc_id, path_res_glosario)
           
           
def word2vec_model(bow, dictionary, path_glosario, path_results):
       #w2v_vector_size = 100
       #model_w2v = models.Word2Vec(sentences=texto_limpio, window=5,
       #                     workers=12, vector_size=w2v_vector_size, min_count=1, seed=50)
       #model_w2v.save(directorio + "/Modelos/word2vec_" + i + ".model")
       #guardar_resultados()
    pass
    
    
#Llamada al modelo de naive bayes
def naivebayes_model(bow_train, bow_test, label_train, dictionary, path_glosario):

    naive_bayes = MultinomialNB()
    naive_bayes.fit(bow_train, label_train)
    predictions = naive_bayes.predict(bow_test)
    
    guardar_resultados()
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
    
    
def create_bow_from_corpus(corpus, dictionary, pathname=None):
    bow = [dictionary.doc2bow(text) for text in corpus]
    if pathname:
        corpora.MmCorpus.serialize(pathname+'/vsm_docs.mm', bow)
    return bow
    
#Realización del pre-proceso de los textos
def process_text(docs, dictionary):
    corpus = clean_docs(docs)
    bow = create_bow_from_corpus(corpus, dictionary)
    return bow


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
                                        
parser.add_argument('-m',
                    "--modelo",
                    type=int,
                    help="Modelo a utilizar para el clasificador. 0 = VSM con tf-idf, 1 = VSM (word2vec), 2 = Naive Bayes")


                    

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
if arguments['modelo']:
    if arguments['modelo'] > 0 and arguments['modelo'] < 3:
        modelo = arguments['modelo']
    else:
        print("ERROR: Introduzca un valor válido mayor que 0 y menor que 2 para un modelo válido")
        exit()

clasificador_documentos(directorio, nmin, rango, modelo)



