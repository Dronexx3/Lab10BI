import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Cargar el archivo de datos
data = pd.read_excel('Data10-1.xlsx')

# Convertir la columna 'Estado' en valores numéricos
data['Estado_num'] = data['Estado'].map({'Alto': 1, 'Bajo': 0})

# Seleccionar características y etiquetas
X = data[['Precio actual', 'Precio final']]
y = data['Estado_num']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar el modelo de árbol de decisión
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# Hacer predicciones
y_pred = tree_model.predict(X_test)

# Calcular la precisión
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')

# Graficar el árbol de decisión
plt.figure(figsize=(12,8))
plot_tree(tree_model, feature_names=['Precio actual', 'Precio final'], class_names=['Bajo', 'Alto'], filled=True)
plt.title('Árbol de Decisión')
plt.show()

