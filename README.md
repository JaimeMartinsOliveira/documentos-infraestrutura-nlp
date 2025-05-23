## 🖼️ Interface da Aplicação

Abaixo um exemplo da interface Streamlit em funcionamento:

![Interface da aplicação](assets/exemplo-interface.png)

# Assistente Inteligente para Gestão de Projetos de Infraestrutura


Este projeto tem como objetivo desenvolver um modelo de NLP capaz de classificar automaticamente trechos de texto relacionados a projetos de infraestrutura pública. Ele foi desenvolvido com foco em demonstrar competências técnicas aplicáveis à área de Inteligência Artificial e Ciência de Dados, com vistas à candidatura à vaga de Engenheiro de IA na Alvarez & Marsal.

## 🎯 Objetivo

O assistente inteligente foi treinado para reconhecer e classificar automaticamente descrições de atividades administrativas, operacionais, técnicas, financeiras e jurídicas presentes em documentos e relatórios de gestão de infraestrutura pública.

## 🧠 Funcionalidades

- **Treinamento supervisionado** com modelo de Machine Learning baseado em vetores TF-IDF e algoritmo Random Forest.
- **Interface interativa com Gradio**, permitindo entrada de texto e classificação em tempo real.
- **Pipeline completo de NLP**: pré-processamento, vetorização, treinamento, avaliação, inferência e deploy.
- **Preparado para escalabilidade**: pronto para ingestão de dados via web scraping e treinamento com bases maiores.

## 🛠️ Tecnologias

- Python 3.10+
- NLTK — para tokenização e stopwords
- scikit-learn — para vetorização e modelo de classificação
- pandas — para manipulação de dados
- Streamlit — para interface interativa
- joblib — para salvar/carregar o modelo


## 🚀 Execução

1. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Treine o modelo**:
   ```bash
   python src/train_model.py
   ```

3. **Execute a inferência com interface Gradio**:
   ```bash
   python src/inference.py
   ```

## 📁 Estrutura

```
# ├── data/                  # Pasta para armazenar os dados
# │   ├── raw/               # Dados brutos
# │   └── processed/         # Dados limpos e preparados
# ├── notebooks/             # Análises e testes exploratórios (Jupyter Notebooks)
# ├── models/                # Modelos treinados salvos
# ├── src/                   # Código-fonte do projeto
# │   ├── __init__.py
# │   ├── data_preprocessing.py
# │   ├── train_model.py
# │   ├── evaluate_model.py
# │   └── inference.py
# ├── app/                   # Interface interativa (Streamlit)
# │   └── app.py
# ├── requirements.txt       # Dependências do projeto
# ├── README.md              # Descrição do projeto
# └── .gitignore
```

## 📩 Contato

**Jaime Martins**  
[jaimemartins.tech](http://jaimemartins.tech) | contato.jaimeMartins@gmail.com | [LinkedIn](https://www.linkedin.com/in/jaime-martins-de-oliveira/)

---

Desenvolvido com foco em excelência e impacto. Vamos transformar dados em decisões.
