# Autores: Ana María Casado y Ana Sanmartin
#
# Este script recoge datos de la línea de comandos introducido por el usuario.

#Dependencias son:
#
#

# IMPORTS 
import os
import argparse
import pandas as pd
import numpy as np
import itertools
from operator import itemgetter
from crear_corpus import pre_procesar_texto
from gensim import corpora, models, similarities
from gensim.corpora.mmcorpus import MmCorpus
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text 
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize.treebank import TreebankWordDetokenizer
from evaluator import get_evaluation

## ESTABLECIDOS POR EL USUARIO ##
directorio =""
nmin = 16
rango = 15
modelo = 0

STOP_WORDS = []
temas = ["Deportes", "Politica", "Salud"]

def clasificador_documentos(directorio, n_min, rango, modelo):
    
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
            f = open(path + i.lower() + str(j) + ".txt","r")
            files = f.read()
           #Se almacenan todos los documentos en una lista para poder procesarlos conjuntamente
            doc += [files]
            doc_id += [i.lower() + str(j)]
            
        #Esto es para train de Naive Bayes, cambiarme porfavor
    if(modelo == 0):
        dictionary = create_dictionary(path_glosario)
    #Pre-procesamiento de los documentos de test    
        bow = process_text(doc, dictionary)    
    else:
        dictionary = None
        bow = doc
    
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
    for doc, score in enumerate(ranking):
        doc_string = doc_id[doc] + ' ' + str(score) + '\n'
        res.write(doc_string)
    res.close()

def guardar_clasificacion(dataframe, doc_id, path_clasificacion):

    np.savetxt(path_clasificacion, dataframe.values, fmt='%s')
    

def guardar_evaluacion(dataframe, path_evaluacion):
    get_evaluation(dataframe, path_evaluacion, temas)
    
    
#####################################
# MODELOS A UTILIZAR EN EL PROYECTO #
#####################################

#Dependiendo del valor de la variable modelo, la función lanzar_clasificador utilizará la llamada al proceso correspondiente    
def lanzar_clasificador(bow, doc_id, dictionary, path_glosario, path_results, m):
    if(m == 0):
        tfidf_model(bow, doc_id, dictionary, path_glosario, path_results)
    elif(m == 1):
        word2vec_model(bow, doc_id, path_glosario, path_results)
    elif(m == 2):
        naivebayes_model(bow, doc_id, path_glosario, path_results)
        
        
def tfidf_model(bow, doc_id, dictionary, path_glosario, path_results):
    tfidf = models.TfidfModel(bow)
    index = similarities.SparseMatrixSimilarity(bow, num_features=len(dictionary))

    final = pd.DataFrame()
    max_values = pd.DataFrame()
    for filename in os.listdir(path_glosario):
        f2 = open(path_glosario + filename, "r")
        glosario = f2.read()
        clean_glosario = wordpunct_tokenize(glosario)
        tfidf_glosario = tfidf[dictionary.doc2bow(clean_glosario)]
        sims = index[tfidf_glosario]
        
        path_res_glosario = path_results+ "/tfidf/tfidf_" + filename
        guardar_resultados(sims, doc_id, path_res_glosario)
        final[cambiar_singlelabel(filename[:-13])] = sims.tolist()

    max_values['Documento'] = doc_id
    max_values['Real'] = cambiar_label([i[:-2] for i in doc_id])
    max_values['Valor'] = final.max(axis = 1)
    max_values['Predicciones'] = final.idxmax(axis=1)
    guardar_clasificacion(max_values, doc_id, path_results + "/Clasificacion/tfidf.txt")
    guardar_evaluacion(max_values, path_results + "/Evaluacion/tfidf_eval.txt")
      
def word2vec_model(bow, doc_id, path_glosario, path_results):
    
    bow = clean_docs(bow)
    w2v_model = models.Word2Vec(bow, size=100, window=5, sg=1, workers=4)

    doc_vec = []
    for doc in bow:
        embeddings = []
        for tok in doc:
            if tok in w2v_model.wv.vocab:
                embeddings.append(w2v_model.wv.word_vec(tok))
            else:
                embeddings.append(np.random.rand(100))

        embeddings = np.mean(embeddings, axis=0)
        doc_vec.append(embeddings)
    
    cs_list = []

    #cv = CountVectorizer(strip_accents = None, preprocessor = None, stop_words = None)
    for filename in os.listdir(path_glosario):
        f2 = open(path_glosario + filename, "r")
        glosario = f2.read()
        glosario = wordpunct_tokenize(glosario)
        
        embedded_glosario=[]
        for tok in glosario:
            if tok in w2v_model.wv.vocab:
                embedded_glosario.append(w2v_model.wv.word_vec(tok))
            else:
                embedded_glosario.append(np.random.rand(100))

        embedded_glosario = np.mean(embedded_glosario, axis=0)

        cs_list=[]
        for i,d_vec in enumerate(doc_vec):
            cs = cosine_similarity(np.array(embedded_glosario).reshape(1,-1),np.array(d_vec).reshape(1, -1))
            cs_list.append((i,cs[0][0]))

        print(cs_list, '\n\n')
        
        path_res_glosario = path_results+ "word2vec/word2vec_" + filename
        guardar_resultados(cs_list, doc_id, path_res_glosario)
        #final[cambiar_singlelabel(filename[:-13])] = sims.tolist()

    
#Llamada al modelo de naive bayes
def naivebayes_model(bow, doc_id, path_glosario, path_results):
    doc_train = []
    label = []
    #final = pd.DataFrame()
    max_values = pd.DataFrame()
    
    for filename in os.listdir(path_glosario):
            f = open(path_glosario+filename,"r")
            files = f.read()
            # Se almacenan todos los documentos en una lista para poder procesarlos conjuntamente
            doc_train += [files]
            label += [filename[:-13]]

    label = cambiar_label(label)

    test = []
    cv = CountVectorizer(strip_accents = None, preprocessor = None, stop_words = None)
    doc_train = cv.fit_transform(doc_train)
    bow = clean_docs(bow)
    
    for s in bow:
        test += ["".join([" "+i if not i.startswith("'") else i for i in s]).strip()]

    bow = cv.transform(test)

    naive_bayes = MultinomialNB()
    naive_bayes.fit(doc_train, label)
    predictions = naive_bayes.predict(bow)
    prediction_probabilities = naive_bayes.predict_proba(bow)
    path_res_glosario = path_results+ "/Naive-Bayes/naivesbayes_results.txt"
    guardar_resultados(predictions, doc_id, path_res_glosario)
    
    final = pd.DataFrame(prediction_probabilities)
    
    max_values['Documento'] = doc_id
    max_values['Real'] = cambiar_label([i[:-2] for i in doc_id])
    max_values['Valor'] = final.max(axis = 1)
    max_values['Predicciones'] = predictions
    guardar_clasificacion(max_values, doc_id, path_results + "/Clasificacion/NaiveBayes.txt")
    guardar_evaluacion(max_values, path_results + "/Evaluacion/NaiveBayes_eval.txt")


###################################
# MÉTODOS PARA PREPROCESAR TEXTOS #
###################################


#THIS IS A TEST, CHANGE LOCATION

#PRUEBA CAMBIAR DE SITIO LUEGO
def cambiar_label(label):
    for j,l in enumerate(label):
        if l.capitalize() == temas[0]:
            label[j] = 0
        elif l.capitalize() == temas[1]:
            label[j] = 1
        elif l.capitalize() == temas[2]:
            label[j] = 2
    return label

def cambiar_singlelabel(l):
    if l == temas[0]:
            l = 0
    elif l == temas[1]:
            l = 1
    elif l == temas[2]:
            l = 2
    return l
    
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
