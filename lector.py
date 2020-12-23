# Autores: Ana Maria Casado y Ana Sanmartin
#
# Este script recoge datos de la linea de comandos introducido por el usuario.

# Paquetes necesarios para el funcionamiento del programa 
import os
import argparse
import pandas as pd
from collections import Counter
from crear_corpus import pre_procesar_texto

## ESTABLECIDOS POR EL USUARIO ##
n = 15
out = os.getcwd()
glosario = 0
modo = 0

def main(directorio, n, modo):
    
    doc = ""
    files = ""
    temas = ["Deportes", "Politica", "Salud"]
    path = ""
    x = 'frecuencia'
    
    # En esta funcion lo que se realizara es obtener los nombres de los documentos presentes en las carpetas
    # llamadas 'Deporte', 'Politica', 'Salud'.
    
    for i in temas:
       doc = ""
       path = directorio + "/Documentos/" + i + "/"
       for j in range(n):
           f = open(path + i.lower() + str(j+1) + ".txt","r")
           files = f.read()
           doc = doc + files

       clean = pre_procesar_texto(doc.lower(), get_modo(modo))
       if(x == 'frecuencia'):
           word_freq = Counter(clean)
           common_words = word_freq.most_common()

       if(x == 'tfidf'):
           pass
           
    # Albergar en un documento los textos correspondientes a cada glosario.
       f2 = open(directorio + "/Pre-Glosario/" + i + ".txt","w+") 
       f2.write(str(common_words))
       f2.close()
    
   
# Funcion que determina el tipo de preprocesamiento del texto 
def get_modo(modo):
   switcher = {
       0 : 'lema',
       1 : 'stem',
       2 : 'none'}
   return switcher.get(modo)
   
######################
# PROGRAMA PRINCIPAL #
######################

# Argparse - Parametros a ser introducidos por el usuario
parser = argparse.ArgumentParser(description="Search by terms")

parser.add_argument('-d',
                    "--directorio",
                    type=str,
                    help="Directorio general donde se encuentran las carpetas con los textos a procesar. El path debe de ser desde la raiz hasta la carpeta Documentos. Ej: .../Document_classification/Documentos")
                    
parser.add_argument('-n',
                    "--n",
                    type=int,
                    help="Numero de documentos a procesar para el glosario")
                    
parser.add_argument('-m',
                    "--modo",
                    type=int,
                    help="Aplicacion de lemmalization, stemmer o nada. Para aplicarlo introducir el numero correspondiente: 0 = lemmalization, 1 = stemmer, 2 = no aplicar nada")

                    

# Parseo de los argumentos
arguments = vars(parser.parse_args())

if arguments['directorio']:
    directorio = arguments['directorio']
else:
    print("ERROR: Porfavor introduzca palabras validas")
    exit()
if arguments['n']:
    if arguments['n'] > 0:
        n = arguments['n']
    else:
        print("ERROR: El valor de N debe ser mayor que 0")
        exit()
if arguments['modo']:
    if arguments['modo'] > 0 and arguments['modo'] < 3:
        modo = arguments['modo']
    else:
        print("ERROR: Introduzca un valor valido entre 0 y 2")
        exit()

main(directorio, n, modo)
