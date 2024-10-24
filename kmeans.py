import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Cargar el archivo de datos
data = pd.read_excel('Data10-1.xlsx')

# Seleccionar las características numéricas
X = data[['Precio actual', 'Precio final']]

# Aplicar K-Means con 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0)
data['Cluster'] = kmeans.fit_predict(X)

# Obtener los centroides
centroides = kmeans.cluster_centers_

# Visualizar los clusters junto con los centroides
plt.scatter(data['Precio actual'], data['Precio final'], c=data['Cluster'], cmap='viridis', label='Datos')
plt.scatter(centroides[:, 0], centroides[:, 1], c='red', s=300, marker='x', label='Centroides')
plt.xlabel('Precio actual')
plt.ylabel('Precio final')
plt.title('K-Means Clustering con Centroides')
plt.legend()
plt.show()

