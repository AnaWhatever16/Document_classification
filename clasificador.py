# Autores: Ana Maria Casado y Ana Sanmartin
#
# Este script se encarga recoger el directorio de trabajo, el número de documentos a procesar, 
# y generar el modelo que se utilizará para la clasificación.

# Paquetes necesarios para el funcionamiento del programa
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

## Variables globales ##
directorio =""
nmin = 16
rango = 15
modelo = 0

STOP_WORDS = []
temas = ["Deportes", "Politica", "Salud"]

def clasificador_documentos(directorio, n_min, rango, modelo):
    """
    Esta función recibe los datos que ha introducido el usuario y llama a otros metodos
    para la creación de los modelos
    
    directorio = directorio de trabajo, path desde la raíz hasta .../Documentation_classification
    n_min = número del documento mas bajo a ser analizado
    rango = número de documentos a ser analizados
    modelo = la clase de modelo que se utilizara en el procesamiento
    """
    
    #Carga de las stop_words
    path = ""
    f = open(directorio +"/stop_words.txt","r")
    sw = f.readlines()
 
    for line in sw: 
        STOP_WORDS.append(line.strip()) 

    
    path_results = directorio + "/Resultados/"
    path_glosario = directorio + "/Glosario/"
    
    doc = []
    doc_id = []
    
    #Para cada tema, se elige cada documento y se almacena todo el texto en una lista 
    #para poder procesarlos conjuntamente
    for i in temas:
        path = directorio + "/Documentos/" + i + "/"
        for j in range(n_min, n_min + rango):
            f = open(path + i.lower() + str(j) + ".txt","r")
            files = f.read()
            doc += [files]
            doc_id += [i.lower() + str(j)]
            
    #Si el modelo es tfidf, se creará un Bag of ords (BoW)
    if(modelo == 0):
        dictionary = create_dictionary(path_glosario)
        #Pre-procesamiento de los documentos de test para generar un BoW  
        bow = process_text(doc, dictionary)    
    else:
        #Si se trata de otro modelo, se pasara el documento completo sin procesar
        dictionary = None
        bow = doc
    
    # Lanzar_clasificador llamara a la funcion del clasificador correspondiente
    lanzar_clasificador(bow, doc_id, dictionary, path_glosario, path_results, modelo)




################################
# ALMACENAMIENTO DE RESULTADOS #
################################

def guardar_resultados(ranking, doc_id, path_results):
    """
    Esta función recibe los resultados de los modelos, los nombres de los documentos y
    un path donde almacenarlos. Se encarga de almacenar las matrices de salida de los resultados
    
    ranking = Matriz con las predicciones de pertenencia de documentos a una clase
    doc_id = nombre del documento almacenado
    path_id = directorio donde se almacenaran los documentos de resultados
    """
    
    res = open(path_results, "w")
    #Para cada fila en la matriz de ranking, escribir una linea en el txt de resultados
    for doc, score in enumerate(ranking):
        doc_string = doc_id[doc] + ' ' + str(score) + '\n'
        res.write(doc_string)
    res.close()


def guardar_clasificacion(dataframe, doc_id, path_clasificacion):
    """
    Esta función recibe las clasificaciones finales realizadas por los modelos, los nombres de los documentos
    y el path donde almacenarlos. Se encarga de almacenar las matrices de salida de las clasificaciones.
    
    dataframe = Matriz con las predicciones finales clasificadas de pertenencia de documentos a una clase
    doc_id = nombre del documento almacenado
    path_clasificacion = directorio donde se almacenaran los documentos de clasificacion
    """
    np.savetxt(path_clasificacion, dataframe.values, fmt='%s')
    

def guardar_evaluacion(dataframe, path_evaluacion):
    """
    Esta función recibe las clasificaciones finales realizadas por los modelos y el path donde se almacenara
    los valores para evaluar la bondad de los modelos. Se encarga de llamar a otro script. 
    
    dataframe = Matriz con las predicciones finales clasificadas de pertenencia de documentos a una clase
    doc_id = nombre del documento almacenado
    path_evaluacion = directorio donde se almacenaran los documentos de evaluacion
    """
    
    get_evaluation(dataframe, path_evaluacion, temas)
    
    
    
    
#####################################
# MODELOS A UTILIZAR EN EL PROYECTO #
#####################################

#Dependiendo del valor de la variable modelo, la funcion lanzar_clasificador utilizara la llamada al proceso correspondiente    
def lanzar_clasificador(bow, doc_id, dictionary, path_glosario, path_results, m):
    """
    Esta función recibe el documento o el BoW, el diccionario, el path donde se encuentran almacenados los glosarios
    el path donde se almacenaran los resultados y el tipo de modelo a crear. Se encarga de llamar al modelo correspondiente.
    
    bow = Textos a ser analizados
    doc_id = nombre del documento almacenado
    dictionary = diccionario que relaciona vectores con las palabras
    path_glosario = carpeta donde se encuentran los glosarios almacenados
    path_results = carpeta donde se almacenaran los resultados
    m = tipo de modelo a utilizar
    """
    
    if(m == 0):
        tfidf_model(bow, doc_id, dictionary, path_glosario, path_results)
    elif(m == 1):
        word2vec_model(bow, doc_id, path_glosario, path_results)
    elif(m == 2):
        naivebayes_model(bow, doc_id, path_glosario, path_results)
        
        
def tfidf_model(bow, doc_id, dictionary, path_glosario, path_results):
    """
    Esta función recibe el documento o el BoW, los nombres de los documentos, el diccionario, el path donde se encuentran almacenados los glosarios
    el path donde se almacenaran los resultados. Se encarga de generar y aplicar el modelo doc2bow con tfidf.
    
    bow = Textos a ser analizados
    doc_id = nombre del documento almacenado
    dictionary = diccionario que relaciona vectores con las palabras
    path_glosario = carpeta donde se encuentran los glosarios almacenados
    path_results = carpeta donde se almacenaran los resultados
    """
    
    #Crea el modelo tfidf
    tfidf = models.TfidfModel(bow)
    index = similarities.SparseMatrixSimilarity(bow, num_features=len(dictionary))

    final = pd.DataFrame()
    max_values = pd.DataFrame()
    
    #Para cada glosario almacenar los resultados en la carpeta tfidf
    for filename in os.listdir(path_glosario):
        f2 = open(path_glosario + filename, "r")
        glosario = f2.read()
        clean_glosario = wordpunct_tokenize(glosario)
        
        #se aplica el modelo doc2bow
        tfidf_glosario = tfidf[dictionary.doc2bow(clean_glosario)]
        sims = index[tfidf_glosario]
        
        path_res_glosario = path_results+ "/tfidf/tfidf_" + filename
        guardar_resultados(sims, doc_id, path_res_glosario)
        final[cambiar_singlelabel(filename[:-13])] = sims.tolist()
    
    #Creacion de una estructura tipo dataframe para estructurar los resultados
    #La matriz contiene el nombre del documento, la clasificacion real, la probabilidad de prediccion
    #y la clasificacion del modelo.
    max_values['Documento'] = doc_id
    max_values['Real'] = cambiar_label([i[:-2] for i in doc_id])
    max_values['Valor'] = final.max(axis = 1)
    max_values['Predicciones'] = final.idxmax(axis=1)
    
    #Se guardan los resultados en clasificacion y en evaluacion
    guardar_clasificacion(max_values, doc_id, path_results + "/Clasificacion/tfidf.txt")
    guardar_evaluacion(max_values, path_results + "/Evaluacion/tfidf_eval.txt")
      

def get_embeddings_from_document(model, g):
    """
    Esta función recibe el modelo y una lista de palabras. Se encarga de calcular la media de los embeddings que existan
    tanto en el modelo como en la lista de palabras
    
    model = modelo creado
    g = lista de palabras en el documento
    """
    
    embeddings = []
    
    #para cada palabra de la lista, si la palabra esta presente en el modelo añadirla a la lista embeddings
    for word in g:
        #print(g)
        if word in model.wv:
            embeddings.append(model.wv[word])
            #print(model.wv)
    
    #calculo de la media
    mean = np.mean(embeddings, axis = 0) if embeddings != [] else np.zeros(10)
    return mean
    
        
def word2vec_model(bow, doc_id, path_glosario, path_results):
    """
    Esta función recibe el documento o el BoW, los nombres de los documentos, el path donde se encuentran almacenados los glosarios
    el path donde se almacenaran los resultados. Se encarga de generar y aplicar el modelo word2vec.
    
    bow = Textos a ser analizados
    doc_id = nombre del documento almacenado
    path_glosario = carpeta donde se encuentran los glosarios almacenados
    path_results = carpeta donde se almacenaran los resultados
    """
    
    #inicializacion de las listas a utilizar
    glosario = []
    glosario_id =[]
    tokens_glosario = []
    
    #limpieza de los textos
    bow = clean_docs(bow)
    
    for filename in os.listdir(path_glosario):
        f2 = open(path_glosario + filename, "r")
        glosario += [f2.read()]
        glosario_id.append(filename)
    
    for g in glosario:
        tokens_glosario += [wordpunct_tokenize(g)]
    
    #creacion del modelo word2vec
    w2v_model = models.Word2Vec(sentences = tokens_glosario, size=10, window=3, sg=1, workers=4, min_count = 1)
        
    query_embedding = [get_embeddings_from_document(w2v_model, g) for g in tokens_glosario]
    vectorized_docs = [get_embeddings_from_document(w2v_model, b) for b in bow]    
    
    final = pd.DataFrame()
    max_values = pd.DataFrame()
    
    #Para cada elemento en los documentos y para cada elemento del glosario aplicar la similaridad del coseno
    for i, g in enumerate(query_embedding):
        similarities = []
        for j, doc in enumerate(vectorized_docs):
            cs = cosine_similarity(np.array(g).reshape(1,-1),np.array(doc).reshape(1, -1))
            similarities.append((cs[0][0]))
        
        #almacenar los resultados
        path_res_glosario = path_results+ "word2vec/word2vec_" + glosario_id[i]
        guardar_resultados(similarities, doc_id, path_res_glosario)
        final[cambiar_singlelabel(glosario_id[i][:-13])] = similarities

    #Creacion de una estructura tipo dataframe para estructurar los resultados
    #La matriz contiene el nombre del documento, la clasificacion real, la probabilidad de prediccion
    #y la clasificacion del modelo.
    max_values['Documento'] = doc_id
    max_values['Real'] = cambiar_label([i[:-2] for i in doc_id])
    max_values['Valor'] = final.max(axis = 1)
    max_values['Predicciones'] = final.idxmax(axis=1)
    
    #Se guardan los resultados en clasificacion y en evaluacion
    guardar_clasificacion(max_values, doc_id, path_results + "/Clasificacion/word2vec.txt")
    guardar_evaluacion(max_values, path_results + "/Evaluacion/word2vec_eval.txt")
    

#Llamada al modelo de naive bayes
def naivebayes_model(bow, doc_id, path_glosario, path_results):
    """
    Esta función recibe el documento o el BoW, los nombres de los documentos, el path donde se encuentran almacenados los glosarios
    el path donde se almacenaran los resultados. Se encarga de generar y aplicar el modelo Naive Bayes.
    
    bow = Textos a ser analizados
    doc_id = nombre del documento almacenado
    path_glosario = carpeta donde se encuentran los glosarios almacenados
    path_results = carpeta donde se almacenaran los resultados
    """
    
    #Inicializacion de las listas
    doc_train = []
    label = []
    max_values = pd.DataFrame()
    
    for filename in os.listdir(path_glosario):
            f = open(path_glosario+filename,"r")
            files = f.read()
            # Se almacenan todos los documentos en una lista para poder procesarlos conjuntamente
            doc_train += [files]
            label += [filename[:-13]]
    
    #Cambiar los nombres de las clases por valores numericos
    label = cambiar_label(label)
    
    #Creacion de CountVectoricer para la transformacion de los datos para poder introducirlos en el modelo
    #El modelo recibe como None, puesto que la limpieza del testo se hara con funciones creadas por nosotras
    test = []
    cv = CountVectorizer(strip_accents = None, preprocessor = None, stop_words = None)
    doc_train = cv.fit_transform(doc_train) #Preparacion de los datos de entrenamiento
    bow = clean_docs(bow) #Limpieza de los textos
    
    #Transformar los token en un string, es lo que espera cv.transform como entrada
    for s in bow:
        test += ["".join([" "+i if not i.startswith("'") else i for i in s]).strip()]

    bow = cv.transform(test) #Preparacion de los datos de test
    
    #Creacion del modelo Naive Bayes e implementacion
    naive_bayes = MultinomialNB()
    naive_bayes.fit(doc_train, label)
    predictions = naive_bayes.predict(bow) 
    prediction_probabilities = naive_bayes.predict_proba(bow) #Prediccion de las clases con probabilidades
    path_res_glosario = path_results+ "/Naive-Bayes/naivesbayes_results.txt"
    guardar_resultados(predictions, doc_id, path_res_glosario)
    
    final = pd.DataFrame(prediction_probabilities)
    
    #Creacion de una estructura tipo dataframe para estructurar los resultados
    #La matriz contiene el nombre del documento, la clasificacion real, la probabilidad de prediccion
    #y la clasificacion del modelo.
    max_values['Documento'] = doc_id
    max_values['Real'] = cambiar_label([i[:-2] for i in doc_id])
    max_values['Valor'] = final.max(axis = 1)
    max_values['Predicciones'] = predictions
    
    #Se guardan los resultados en clasificacion y en evaluacion
    guardar_clasificacion(max_values, doc_id, path_results + "/Clasificacion/NaiveBayes.txt")
    guardar_evaluacion(max_values, path_results + "/Evaluacion/NaiveBayes_eval.txt")




###################################
# METODOS PARA PREPROCESAR TEXTOS #
###################################


def cambiar_label(label):
    """
    Esta función recibe una lista de palabras. Se encarga de transformar los elementos de la lista en numeros 
    representando una clase.
    
    label = Lista de palabras
    """
    
    for j,l in enumerate(label):
        if l.capitalize() == temas[0]:
            label[j] = 0
        elif l.capitalize() == temas[1]:
            label[j] = 1
        elif l.capitalize() == temas[2]:
            label[j] = 2
    return label

def cambiar_singlelabel(l):
    """
    Esta función recibe una palabras. Se encarga de transformar una palabra en im numero 
    representando una clase.
    
    label = palabras
    """
    
    if l == temas[0]:
            l = 0
    elif l == temas[1]:
            l = 1
    elif l == temas[2]:
            l = 2
    return l
    


def clean_docs(docs):
    """
    Esta función recibe una lista con documentos. Se encarga de limpiar una lista de texto.
    
    docs = lista con el texto de los documentos
    """
    
    #Inicializacion de variables
    stemmer = PorterStemmer()
    final = []
    
    #Para cada documento en la lista de documentos aplicar: tokenizacion, eliminar stop words,
    #eliminar si tienen menos de 2 elementos y aplicar radicalizacion
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



def create_dictionary(path_glosario, pathname= None):
    """
    Esta función recibe el path donde se encuentran los glosarios. Se encarga de convertir los textos en un diccionario 
    y los almacena en corpus 
    
    path_glosario = path donde se encuentra almacenado el glosario
    """
    #se crea un diccionario de corpora
    dictionary = corpora.Dictionary()
    
    #Para cada palabra del diccionario se separa en tokens
    for filename in os.listdir(path_glosario):
        f2 = open(path_glosario + filename, "r")
        glosario = f2.read()
        tokens = [word for word in glosario.split()]
        dictionary.add_documents([tokens])
    
    #Guardar el diccionario en memoria
    if pathname:
        dictionary.save(pathname+"/corpus.dict")
    return dictionary
    
    
def create_bow_from_corpus(corpus, dictionary, pathname=None):
    """
    Esta función recibe un corpus creado a partir del glosario y un diccionario a partir del glosario.
    Se encarga de transformar un diccionario en una Bag of Words (BoW)
    
    corpus = lista de documentos
    dictionary = diccionario
    """
    
    #Se crea el Bag of Words mediante la funcion de doc2bow
    bow = [dictionary.doc2bow(text) for text in corpus]
    
    #Guardar el bow en memoria
    if pathname:
        corpora.MmCorpus.serialize(pathname+'/vsm_docs.mm', bow)
    return bow
    
    
#Realizacion del pre-proceso de los textos
def process_text(docs, dictionary):
    """
    Esta función recibe una la lista de documentos y el diccionario creado por corpora
    Se encarga de procesar los documentos y crear un bow del corpus a partir de los documentos y el diccionario llamando
    a la funcion create_bow_from_corpus.
    
    docs = lista de documentos
    dictionary = diccionario
    """
    
    corpus = clean_docs(docs)
    bow = create_bow_from_corpus(corpus, dictionary)
    return bow


###########################################
# METODOS PARA OBTENER DATOS POR PANTALLA #
###########################################

# Argparse - Parametros a ser introducidos por el usuario
parser = argparse.ArgumentParser(description="Search by terms")

parser.add_argument('-d',
                    "--directorio",
                    type=str,
                    help="Directorio general donde se encuentran las carpetas con los textos a procesar. El path debe de ser desde la raiz hasta la carpeta Documentos. Ej: .../Document_classification/Documentos")
                    
parser.add_argument('-n',
                    "--nmin",
                    type=int,
                    help="Posicion a partir de la cual se utilizaran los documentos (Por ejemplo, a partir del documento 16)")
                    
parser.add_argument('-r',
                    "--rango",
                    type=int,
                    help="Numero de documentos totales a utilizar para la clasificacion de test (Por ejemplo si deseamos 15 documentos, del 16 al 30)")
                                        
parser.add_argument('-m',
                    "--modelo",
                    type=int,
                    help="Modelo a utilizar para el clasificador. 0 = VSM con tf-idf, 1 = VSM (word2vec), 2 = Naive Bayes")


# Parseo de los argumentos
arguments = vars(parser.parse_args())

if arguments['directorio']:
    directorio = arguments['directorio']
else:
    print("ERROR: Porfavor introduzca palabras validas para el directorio")
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
        print("ERROR: Introduzca un valor valido mayor que 0")
        exit()
if arguments['modelo']:
    if arguments['modelo'] > 0 and arguments['modelo'] < 3:
        modelo = arguments['modelo']
    else:
        print("ERROR: Introduzca un valor valido mayor que 0 y menor que 2 para un modelo valido")
        exit()

clasificador_documentos(directorio, nmin, rango, modelo)
