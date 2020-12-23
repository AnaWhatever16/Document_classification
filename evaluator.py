# Autores: Ana Maria Casado y Ana Sanmartin
#
# Este script calcula la precision, exactitud y sensibilidad/exhaustividad de los resultados obtenidos.
# Ademas, los imprime por pantalla, los guarda un documento y crea una matriz de confusion que muestra graficamente los resultados.

# Paquetes necesarios para el funcionamiento del programa
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def get_evaluation(results, path_evaluacion, temas):

    print('Accuracy score: ' + str(accuracy_score(results['Real'], results['Predicciones']))) #Calculo de la exactitud
    print('Precision score: ' + str(precision_score(results['Real'], results['Predicciones'], average = 'macro'))) #Calculo de la precision
    print('Recall score: ' + str( recall_score(results['Real'], results['Predicciones'], average = 'macro'))) #Calculo de la sensibilidad/exhaustividad
    
    f = open(path_evaluacion, "w")
    f.write('Accuracy score: ' + str(accuracy_score(results['Real'], results['Predicciones'])) + '\n')
    f.write('Precision score: ' + str(precision_score(results['Real'], results['Predicciones'], average = 'macro')) + '\n')
    f.write('Recall score: ' + str( recall_score(results['Real'], results['Predicciones'], average = 'macro')) + '\n')
    f.close()
    
    
    cm = confusion_matrix(results['Real'], results['Predicciones']) #Creacion de la matriz de confusion
    sns.heatmap(cm, square=True, annot=True, cmap='Oranges', cbar=False,
    xticklabels = temas, yticklabels=temas) # Creacion del grafico que muestra los resultados de la matriz de confusion 
    plt.ylabel('Clases reales')
    plt.xlabel('Predicciones de clases')
    plt.show()

