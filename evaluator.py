# Comprobar los resultados y calcular la precisión y exhaustividad de los métodos.
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def get_evaluation(results, path_evaluacion, temas):

    print('Accuracy score: ' + str(accuracy_score(results['Real'], results['Predicciones'])))
    print('Precision score: ' + str(precision_score(results['Real'], results['Predicciones'], average = 'macro')))
    print('Recall score: ' + str( recall_score(results['Real'], results['Predicciones'], average = 'macro')))
    
    f = open(path_evaluacion, "w")
    f.write('Accuracy score: ' + str(accuracy_score(results['Real'], results['Predicciones'])) + '\n')
    f.write('Precision score: ' + str(precision_score(results['Real'], results['Predicciones'], average = 'macro')) + '\n')
    f.write('Recall score: ' + str( recall_score(results['Real'], results['Predicciones'], average = 'macro')) + '\n')
    f.close()
    
    
    cm = confusion_matrix(results['Real'], results['Predicciones'])
    sns.heatmap(cm, square=True, annot=True, cmap='RdBu', cbar=False,
    xticklabels = temas, yticklabels=temas)
    plt.ylabel('Clases reales')
    plt.xlabel('Predicciones de clases')
    plt.show()

