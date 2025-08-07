import math
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report



file_path = "/home/agus3112/Documents/Facu/4to/ia2-uncuyo-2025/LogisticRegression/Marketingcampaigns.csv"
data_frame = pd.read_csv(file_path)


'''
purchase_column = data_frame['Purchased']

#Comparacion entre personas que compraron y no compraron

print("Cantidad de compras:", purchase_column.sum())
print("Cantidad de no compras:", len(purchase_column) - purchase_column.sum())
'''



'''
data_frame['AgeGroup'] = pd.cut(data_frame['Age'], bins=[10, 20, 30, 40, 50, 60], right=False)

# Filtrar solo los que compraron
compras = data_frame[data_frame['Purchased'] == 1]

# Contar compras por grupo de edad
compras_por_edad = compras['AgeGroup'].value_counts().sort_index()


compras_por_edad.plot(kind='bar')
plt.ylabel('Cantidad de compras')
plt.xlabel('Grupo etario')
plt.title('Compras por grupo de edad')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
'''



'''
# Relacion entre clic en el email y compras
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
# Relacion entre localidad y compras

tabla = pd.crosstab(data_frame['Location'], data_frame['Purchased'])


tabla.plot(kind='bar', stacked=True)
plt.title('Relación entre localidad y compras')
plt.xlabel('Localidad')
plt.ylabel('Cantidad de personas')
plt.legend(title='Compró (0 = No, 1 = Sí)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
'''


'''
# Relacion entre descuento y compras
tabla = pd.crosstab(data_frame['Discount offered'], data_frame['Purchased'])

# Gráfico
tabla.plot(kind='bar', stacked=True)
plt.title('¿El descuento influye en la compra?')
plt.xlabel('¿Se ofreció descuento? (0 = No, 1 = Sí)')
plt.ylabel('Cantidad de personas')
plt.legend(title='Compró (0 = No, 1 = Sí)')
plt.tight_layout()
plt.show()
'''





'''
# Relacion entre Genero y compras
tabla = pd.crosstab(data_frame['Gender'], data_frame['Purchased'])


tabla.plot(kind='bar', stacked=True)
plt.title('Relación entre género y compras')
plt.xlabel('Género (0 = Femenino, 1 = Masculino)')
plt.ylabel('Cantidad de personas')
plt.legend(title='Compró (0 = No, 1 = Sí)')
plt.tight_layout()
plt.show()
'''



'''
# Relacion entre visitas a la pagina del producto y compras
sb.boxplot(x='Purchased', y='Product page visit', data=data_frame)
plt.title('Visitas a la página según si compró o no')
plt.xlabel('Compró (0 = No, 1 = Sí)')
plt.ylabel('Visitas a la página del producto')
plt.tight_layout()
plt.show()
'''

## Modelo Predictivo

# Variables predictoras
X = data_frame[['Age', 'Gender', 'Email Opened', 'Email Clicked', 'Product page visit', 'Discount offered']]

# Variable objetivo
y = data_frame['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de regresión logística
modelo = LogisticRegression()

# Entrenar el modelo
modelo.fit(X_train, y_train)

'''
# peso de los coeficientes
print("Intercepto (w0):", modelo.intercept_)
print("Coeficientes (w1, w2, ..., wn):", modelo.coef_)
print("Atributos:", X_train.columns)
'''


y_pred = modelo.predict(X_test)
y_prob = modelo.predict_proba(X_test)[:, 1]


print(":", y_prob)
# Ver resultados
#print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# También un reporte completo:
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))




