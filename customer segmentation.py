```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import cv2
```

```python
df = pd.read_csv("customers.csv")
df.head()
```

```python
df.info()
df.describe()
df.isnull().sum()
```

```python
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]
```

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

```python
inertia = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(2, 11), inertia, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()
```

```python
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
```

```python
sns.pairplot(df, hue='Cluster', palette='tab10')
plt.show()
```

```python
plt.figure(figsize=(8,5))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, palette='tab10')
plt.title("Customer Segments by Income and Spending Score")
plt.show()
```

```python
silhouette = silhouette_score(X_scaled, df['Cluster'])
print("Silhouette Score:", silhouette)
```

```python
model = Sequential([
    Dense(16, activation='relu', input_shape=(3,)),
    Dense(8, activation='relu'),
    Dense(5, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

```python
y = df['Cluster']
model.fit(X_scaled, y, epochs=50, batch_size=8, verbose=0)
```

```python
test_data = np.array([[35, 60, 80]])
test_scaled = scaler.transform(test_data)
prediction = model.predict(test_scaled)
print("Predicted Cluster (Deep Learning):", np.argmax(prediction))
```

```python
image = np.zeros((200, 200, 3), dtype=np.uint8)
cv2.circle(image, (100, 100), 50, (255, 0, 0), -1)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)
plt.imshow(edges, cmap='gray')
plt.title("OpenCV Image Processing Example")
plt.show()
```
