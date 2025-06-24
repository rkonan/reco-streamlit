FROM python:3.10-slim

# Créer le répertoire de travail
WORKDIR /app

# Copier le fichier de dépendances et installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY . .

# Lancer l'application Streamlit
CMD ["streamlit", "run", "app/main.py", "--server.address=0.0.0.0", "--server.port=8052"]
