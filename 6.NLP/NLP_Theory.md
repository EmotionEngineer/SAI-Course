# 💬 Natural Language Processing Basics

Natural Language Processing (NLP) is a field of AI focused on enabling machines to understand, interpret, and generate human language. NLP powers applications from chatbots and translation systems to more sophisticated tasks like sentiment analysis and language modeling.

---

## 🔍 What is Natural Language Processing?

NLP combines elements of **linguistics**, **computer science**, and **machine learning** to process and analyze human language. It transforms text into structured information that machines can understand, enabling them to perform tasks such as **text classification**, **machine translation**, and **question answering**.

---

## 📏 Key Concepts in NLP

### 1. **Text Representation**

Raw text cannot be directly fed into machine learning algorithms. NLP begins with converting text into numerical representations that preserve meaning and context. Popular representations include:

   - **Bag of Words (BoW)**: Represents text as a set of individual words, ignoring order but counting occurrences.
   - **TF-IDF (Term Frequency-Inverse Document Frequency)**: Weights words based on their frequency in a document relative to their frequency across all documents, emphasizing unique words.
   
  **Formula** for TF-IDF for a word $w$ in a document $d$ from a corpus $D$:
  
  $$
  \text{TF-IDF}(w, d, D) = \text{TF}(w, d) \times \text{IDF}(w, D)
  $$
  
  Where:
  
  - **Term Frequency (TF)** measures how often a word appears in a document:
  
  $$
  \text{TF}(w, d) = \frac{\text{Frequency of } w \text{ in } d}{\text{Total words in } d}
  $$
  
  - **Inverse Document Frequency (IDF)** reduces the weight of commonly used words:
  
  $$
  \text{IDF}(w, D) = \log\left(\frac{|D|}{1 + |\{d \in D : w \in d\}|}\right)
  $$
  
   - **Word Embeddings (e.g., Word2Vec, GloVe)**: Embeddings capture context and similarity by mapping words to vectors of real numbers in high-dimensional space.

   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer

   documents = ["I love NLP.", "NLP is amazing."]
   vectorizer = TfidfVectorizer()
   tfidf_matrix = vectorizer.fit_transform(documents)
   ```

### 2. **Tokenization**

Tokenization splits text into smaller units, such as words or subwords, making it easier to analyze. Tokenizers may remove punctuation, lowercase words, or even split words further into morphemes.

- **Word Tokenization**: Breaks text into individual words.
- **Subword Tokenization**: Breaks text into meaningful word parts (used in BERT and GPT models).

```python
from nltk.tokenize import word_tokenize
text = "Tokenization is essential in NLP."
tokens = word_tokenize(text)
```

### 3. **Stop Words and Stemming/Lemmatization**

   - **Stop Words**: Words like "the," "is," and "and" are often removed because they carry little unique information.
   - **Stemming and Lemmatization**: Reduce words to their root form to group similar words together.
   
   ```python
   from nltk.corpus import stopwords
   from nltk.stem import PorterStemmer

   words = ["running", "runner", "ran"]
   stemmer = PorterStemmer()
   stems = [stemmer.stem(word) for word in words]
   ```

### 4. **Named Entity Recognition (NER)**

NER is a technique used to identify and classify **proper nouns** in text, like names of people, organizations, dates, and locations. NER models classify text spans into categories such as **Person**, **Organization**, **Location**, etc.

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple was founded by Steve Jobs.")
entities = [(ent.text, ent.label_) for ent in doc.ents]
```

---

## 🧠 Neural Networks in NLP

### Recurrent Neural Networks (RNNs)

**Recurrent Neural Networks** are designed for sequential data. In RNNs, information cycles through layers, enabling the network to remember previous inputs. However, RNNs struggle with long-term dependencies, which has led to the development of **Long Short-Term Memory (LSTM)** and **Gated Recurrent Unit (GRU)** networks.

#### RNN Formula

For an RNN layer, each output $h_t$ at time step $t$ is calculated as:

$$
h_t = f(W \cdot h_{t-1} + U \cdot x_t + b)
$$

Where:

- $h_t$: Hidden state at time $t$,
- $W$: Weight matrix for hidden state,
- $U$: Weight matrix for input,
- $x_t$: Input at time $t$,
- $b$: Bias, and
- $f$: Activation function, usually tanh or ReLU.

### Transformer Networks

Transformers, like **BERT** and **GPT**, use **self-attention** mechanisms to process all tokens in parallel, capturing long-range dependencies without the sequential limitations of RNNs.

#### Self-Attention Mechanism

In self-attention, each word in a sequence interacts with every other word, weighted by a similarity score. Given query ($Q$), key ($K$), and value ($V$) matrices, the attention score is computed as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

Where:

- $Q$, $K$, and $V$: Representations of the input sequence,
- $d_k$: Dimension of $K$, used to scale the dot product.

---

## 🔄 NLP Model Workflow

The process of building an NLP model typically follows these steps:

1. **Data Collection**: Gather and preprocess text data (e.g., product reviews for sentiment analysis).
2. **Preprocessing**: Clean the text by tokenizing, stemming/lemmatizing, and vectorizing it.
3. **Feature Extraction**: Represent text with embeddings or TF-IDF vectors.
4. **Model Training**: Train the model (e.g., RNN, Transformer) to learn from labeled data.
5. **Evaluation**: Test on unseen data, adjusting hyperparameters for better accuracy.
6. **Deployment**: Integrate the model into applications (e.g., a chatbot or sentiment analysis tool).

---

## 🤖 Implementing a Simple NLP Model in Python (using Keras)

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data
sentences = ["I love NLP", "NLP is amazing", "I enjoy learning AI", "I hate AI"]
labels = [1, 1, 1, 0]  # Sentiment (1 = Positive)

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, maxlen=5)

# Define a simple LSTM model
model = Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=5),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

# Compile and summarize the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

---

## 🔥 Key NLP Models

1. **BERT (Bidirectional Encoder Representations from Transformers)**: Uses self-attention to capture word relationships in both directions, useful for understanding context in sentences.
2. **GPT (Generative Pre-trained Transformer)**: A generative model effective for text generation, completion, and chatbot applications.
3. **Word2Vec**: Learns word embeddings using contexts of words in a sentence, enhancing semantic understanding.

---

## 📐 NLP Applications

NLP has a wide array of applications across industries:

- **Sentiment Analysis**: Classifies text by emotion (e.g., positive, negative).
- **Machine Translation**: Translates text between languages.
- **Chatbots and Virtual Assistants**: Enables conversational interfaces.
- **Summarization**: Creates summaries from lengthy texts.
- **Named Entity Recognition**: Identifies specific names, dates, and other entities.

---

## 🧩 NLP in Summary

NLP enables machines to comprehend and generate language, unlocking applications in everything from customer support to content creation. By using models from RNNs to Transformers, we can process and make sense of vast amounts of textual data.

### Quick Recap:
- **Text Representation** is key to transforming raw text for analysis.
- **Tokenization and Feature Extraction** prepare text for machine learning.
- **NLP Models** range from traditional RNNs to state-of-the-art Transformers.
- **Applications** include translation, chatbots, summarization, and more.
