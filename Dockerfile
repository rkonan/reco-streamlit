# Utilise une image Python 3.10 officielle
FROM python:3.10-slim

# Installe les dépendances système pour OpenCV (optionnel)
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copie les fichiers dans le container
WORKDIR /app
COPY . /app

# Installe les dépendances Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose le port 8501 (par défaut Streamlit)
EXPOSE 8501

# Commande de démarrage
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
