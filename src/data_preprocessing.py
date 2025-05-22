import os
import re
import nltk
import string
import pandas as pd
from typing import List, Union
from sklearn.feature_extraction.text import TfidfVectorizer

# Baixar recursos do NLTK na primeira vez
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words('portuguese'))

def read_text_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def clean_text(text: str) -> str:
    # Remove números, pontuação, múltiplos espaços e deixa tudo lowercase
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text: str) -> str:
    tokens = word_tokenize(text, language='portuguese')
    filtered_tokens = [t for t in tokens if t not in STOPWORDS]
    return ' '.join(filtered_tokens)

def preprocess_text(text: str) -> str:
    text = clean_text(text)
    text = remove_stopwords(text)
    return text

def preprocess_corpus(corpus: List[str]) -> List[str]:
    return [preprocess_text(doc) for doc in corpus]

def load_dataset_csv(file_path: str, text_column: str, label_column: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df = df[[text_column, label_column]].dropna()
    return df

def vectorize_text(corpus: List[str], max_features: int = 5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

if __name__ == "__main__":
    # Exemplo rápido de uso:
    sample_text = "Este é um exemplo de texto técnico da área de infraestrutura, com números 123 e símbolos %$#."
    processed = preprocess_text(sample_text)
    print("Texto original:", sample_text)
    print("Texto preprocessado:", processed)
