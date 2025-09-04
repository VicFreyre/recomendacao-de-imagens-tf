# 🔎 Sistema de Recomendação por Imagens

Este projeto implementa um **sistema de recomendação baseado em imagens**, utilizando **Deep Learning** para extrair características visuais e recomendar itens semelhantes.  
O dataset usado é o **Fashion-MNIST**, que contém imagens de roupas e acessórios em escala de cinza.
<img width="1070" height="216" alt="image" src="https://github.com/user-attachments/assets/053ca6c9-96f2-4bae-85d4-557d5bffe297" />

---

## 📌 Objetivo
Criar um modelo capaz de recomendar imagens semelhantes com base em sua **aparência visual** (formato, textura, cor), sem depender de atributos textuais como preço, marca ou descrição.

---

## ⚙️ Tecnologias Usadas
- Python 
- TensorFlow / Keras 
- Scikit-learn  
- Matplotlib   

---

## 📂 Estrutura
- `Treinamento` → Rede neural convolucional (CNN) para extrair **embeddings** das imagens.  
- `Recomendação` → Busca pelas imagens mais similares usando **similaridade do cosseno**.  
- `Visualização` → Exibe a imagem consultada e as recomendações lado a lado.  

---
**negrito**
## 🚀 Como Executar no Google Colab
1. Abra o Colab [aqui](https://colab.research.google.com/drive/1wUi_Q0Kd1EJHw1-hEJ1jt8lg_Ae8t9Tc?authuser=0#scrollTo=emuS_QEwcXh1).  
2. Copie e cole o código do projeto.  
3. Execute todas as células.  
4. Para testar, use:  

```python
idx = np.random.randint(0, len(x_test))
recomendar(idx, top_n=5)
