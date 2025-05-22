import os

# Diretórios do projeto
dirs = [
    "data/raw",
    "data/processed",
    "notebooks",
    "models",
    "src",
    "app"
]

# Arquivos iniciais (com conteúdo opcional)
files = {
    "src/__init__.py": "",
    "src/data_preprocessing.py": "# Funções de pré-processamento\n",
    "src/train_model.py": "# Script para treinamento do modelo\n",
    "src/evaluate_model.py": "# Script para avaliação do modelo\n",
    "src/inference.py": "# Script para fazer inferência com o modelo treinado\n",
    "app/app.py": "# Interface com Streamlit\n",
    "requirements.txt": "# Adicione suas dependências aqui\n",
    "README.md": "# Classificador de Documentos Técnicos de Infraestrutura\n\nProjeto de NLP com Transformers para classificar documentos técnicos por setor.",
    ".gitignore": "*.pyc\n__pycache__/\n.env\nmodels/\ndata/\n"
}

# Cria os diretórios
for d in dirs:
    os.makedirs(d, exist_ok=True)

# Cria os arquivos
for path, content in files.items():
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

print("✅ Estrutura de projeto criada com sucesso!")