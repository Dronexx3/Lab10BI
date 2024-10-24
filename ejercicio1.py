import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Generar datos simulados
np.random.seed(1)
X1 = np.random.randn(100)  # Característica 1
X2 = np.random.randn(100)  # Característica 2
y = np.where(X1 + X2 > 0, 1, -1)  # Etiquetas

# Crear un DataFrame
datos = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})

# Dividir los datos en entrenamiento y prueba
X = datos[['X1', 'X2']]
y = datos['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo SVM
modelo_svm = SVC(kernel='linear', C=10)
modelo_svm.fit(X_train, y_train)

# Función para visualizar los resultados
def plot_svm_decision_boundary(modelo, X, y):
    plt.figure(figsize=(10, 6))
    plt.scatter(X['X1'], X['X2'], c=y, cmap='coolwarm', s=50, edgecolors='k')

    # Crear una malla para graficar la frontera de decisión
    xlim = plt.xlim()
    ylim = plt.ylim()

    xx = np.linspace(xlim[0], xlim[1], 100)
    yy = np.linspace(ylim[0], ylim[1], 100)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = modelo.decision_function(xy).reshape(XX.shape)

    plt.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'])

    plt.title('Clasificación SVM con Kernel Lineal')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()

# Llamar a la función para graficar
plot_svm_decision_boundary(modelo_svm, X_train, y_train)

# Evaluar el modelo
accuracy = modelo_svm.score(X_test, y_test)
print(f'Precisión del modelo: {accuracy:.2f}')