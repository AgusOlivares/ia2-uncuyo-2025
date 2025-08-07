import math
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


file_path = "/home/agus3112/Documents/Facu/4to/ia2-uncuyo-2025/LogisticRegression/Marketingcampaigns.csv"
data_frame = pd.read_csv(file_path)

purchase_column = data_frame['Purchased']

#Comparacion entre personas que compraron y no compraron

print("Cantidad de compras:", purchase_column.sum())
print("Cantidad de no compras:", len(purchase_column) - purchase_column.sum())

'''
data_frame['AgeGroup'] = pd.cut(data_frame['Age'], bins=[10, 20, 30, 40, 50, 60], right=False)

# Filtrar solo los que compraron
compras = data_frame[data_frame['Purchased'] == 1]

# Contar compras por grupo de edad
compras_por_edad = compras['AgeGroup'].value_counts().sort_index()

# Graficar
compras_por_edad.plot(kind='bar')
plt.ylabel('Cantidad de compras')
plt.xlabel('Grupo etario')
plt.title('Compras por grupo de edad')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
'''

'''
tabla = pd.crosstab(data_frame['Email Clicked'], data_frame['Purchased'])

# Graficar
tabla.plot(kind='bar', stacked=True)
plt.title('Relación entre clic en el email y compras')
plt.xlabel('¿Hizo clic en el email? (0 = No, 1 = Sí)')
plt.ylabel('Cantidad de personas')
plt.legend(title='Compró (0 = No, 1 = Sí)')
plt.tight_layout()
plt.show()

'''


'''
tabla = pd.crosstab(data_frame['Location'], data_frame['Purchased'])

# Graficar
tabla.plot(kind='bar', stacked=True)
plt.title('Relación entre localidad y compras')
plt.xlabel('Localidad')
plt.ylabel('Cantidad de personas')
plt.legend(title='Compró (0 = No, 1 = Sí)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
'''