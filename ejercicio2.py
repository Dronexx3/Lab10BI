import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Cargar los datos del archivo Excel
file_path = 'Data10-1.xlsx'  # Asegúrate de poner la ruta correcta del archivo
data = pd.read_excel(file_path, sheet_name='Hoja1')

# Seleccionar las columnas relevantes para el modelo
X = data[['Precio actual', 'Precio final']].values
y = data['Estado']

# Convertir la columna 'Estado' a valores numéricos (Alto: 1, Bajo: 0)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Crear y entrenar el modelo SVM con kernel lineal
modelo_svm = svm.SVC(kernel='linear')
modelo_svm.fit(X_train, y_train)

# Hacer predicciones con el conjunto de prueba
y_pred = modelo_svm.predict(X_test)

# Calcular la precisión del modelo
precision = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {precision * 100:.2f}%')

# Función para graficar los datos y el hiperplano de separación
def plot_svm(modelo, X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    
    # Dibujar el hiperplano
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Crear una malla para graficar el hiperplano
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = modelo.decision_function(xy).reshape(XX.shape)

    # Dibujar el hiperplano
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # Dibujar los vectores de soporte
    ax.scatter(modelo.support_vectors_[:, 0], modelo.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')

# Graficar los datos y el hiperplano
plot_svm(modelo_svm, X_train, y_train)
plt.title("SVM - Precio actual vs Precio final")
plt.xlabel("Precio actual")
plt.ylabel("Precio final")
plt.show()
