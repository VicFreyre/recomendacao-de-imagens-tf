# ============================================
# 🔹 Sistema de Recomendação por Imagens com Dataset Pronto
# Dataset: Fashion-MNIST (roupas, sapatos, bolsas, camisetas)
# ============================================

!pip install tensorflow matplotlib scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics.pairwise import cosine_similarity

# ============================================
# 🔹 Carregar Dataset Fashion-MNIST
# ============================================
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalizar e expandir dimensões (para CNN)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Classes do Fashion-MNIST
classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ============================================
# 🔹 Criar uma rede CNN simples para extrair embeddings
# ============================================
inputs = Input(shape=(28,28,1))
x = Conv2D(32, (3,3), activation="relu")(inputs)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64, (3,3), activation="relu")(x)
x = MaxPooling2D((2,2))(x)
x = Flatten()(x)
embeddings = Dense(64, activation="relu")(x)  # Vetor de características
outputs = Dense(10, activation="softmax")(embeddings)

model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("⏳ Treinando modelo...")
model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1)

# Criar modelo que retorna só os embeddings
feature_extractor = Model(inputs, embeddings)

# Extrair features de todo o conjunto de teste
features = feature_extractor.predict(x_test)

# ============================================
# 🔹 Função de Recomendação
# ============================================
def recomendar(idx, top_n=5):
    query_feat = features[idx].reshape(1, -1)
    sims = cosine_similarity(query_feat, features).flatten()
    idxs = sims.argsort()[-top_n-1:][::-1]  # pega os mais similares
    
    plt.figure(figsize=(12,3))
    plt.subplot(1, top_n+1, 1)
    plt.imshow(x_test[idx].reshape(28,28), cmap="gray")
    plt.title("Consulta\n" + classes[y_test[idx]])
    plt.axis("off")

    for i, sim_idx in enumerate(idxs[1:top_n+1]):
        plt.subplot(1, top_n+1, i+2)
        plt.imshow(x_test[sim_idx].reshape(28,28), cmap="gray")
        plt.title(f"Similar {i+1}\n{classes[y_test[sim_idx]]}")
        plt.axis("off")
    plt.show()

# ============================================
# 🔹 Exemplo de uso
# ============================================
# Teste com uma imagem aleatória
idx = np.random.randint(0, len(x_test))
recomendar(idx, top_n=5)
