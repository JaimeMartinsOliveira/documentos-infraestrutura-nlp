import pickle
from typing import List

def load_model(model_path: str):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_vectorizer(vectorizer_path: str):
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer

def preprocess_text(text: str, preprocess_func) -> str:
    return preprocess_func(text)

def predict(texts: List[str], model, vectorizer, preprocess_func) -> List:
    processed_texts = [preprocess_text(t, preprocess_func) for t in texts]
    X = vectorizer.transform(processed_texts)
    preds = model.predict(X)
    return preds

if __name__ == "__main__":
    from data_preprocessing import preprocess_text

    model = load_model('models/model.pkl')
    vectorizer = load_vectorizer('models/vectorizer.pkl')

    samples = [
        "Projeto de construção de rodovia na região sudeste.",
        "Análise de desempenho de rede de telecomunicações."
    ]

    predictions = predict(samples, model, vectorizer, preprocess_text)
    for text, pred in zip(samples, predictions):
        print(f"Texto: {text}\nPredição: {pred}\n")
