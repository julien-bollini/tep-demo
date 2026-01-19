# Utilisation d'une image Python légère
FROM python:3.11-slim

# Définition du répertoire de travail dans le conteneur
WORKDIR /app

# Installation des dépendances directement
# (Tu pourras plus tard copier un requirements.txt ici)
RUN pip install --no-cache-dir kagglehub

# Copie du script de téléchargement dans le conteneur
COPY data_pipeline/src/download_data.py /app/data_pipeline/src/download_data.py

# Création du dossier de données (qui servira de point de montage)
RUN mkdir -p /app/data

# Commande par défaut pour lancer le script
CMD ["python", "/app/data_pipeline/src/download_data.py"]
