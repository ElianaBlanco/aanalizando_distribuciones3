import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px

# Eliminamos la columna DEATH_EVENT y la columna de categoría de edad
X = df.drop(columns=['DEATH_EVENT', 'categoria_edad']).values

# Vector con la columna DEATH_EVENT
y = df['DEATH_EVENT'].values

# Reducción de dimensionalidad a 3D
X_embedded = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=3).fit_transform(X)

# Crear un DataFrame con los datos reducidos y la columna objetivo
df_embedded = pd.DataFrame({'x': X_embedded[:, 0], 'y': X_embedded[:, 1], 'z': X_embedded[:, 2], 'DEATH_EVENT': y})

# Crear el gráfico 3D
fig = px.scatter_3d(df_embedded, x='x', y='y', z='z', color='DEATH_EVENT', title='Gráfico de dispersión 3D - DEATH_EVENT')
fig.show()
