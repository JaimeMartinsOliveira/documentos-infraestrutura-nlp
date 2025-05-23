import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from data_preprocessing import preprocess_corpus, load_dataset_csv

def train():
    df = load_dataset_csv('data/processed/train.csv', text_column='text', label_column='label')

    texts = preprocess_corpus(df['text'].tolist())
    labels = df['label'].tolist()

    X_train_texts, X_val_texts, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42 )#stratify=labels)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(X_train_texts)
    X_val = vectorizer.transform(X_val_texts)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    print("Avaliação no conjunto de validação:")
    print(classification_report(y_val, y_pred))

    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    print("Treinamento concluído e modelos salvos em /models")


if __name__ == "__main__":
    train()
