# [cite_start]Utilisation d'une image Python légère [cite: 2]
FROM python:3.11-slim

WORKDIR /app

# [cite_start]Installation des dépendances système [cite: 2]
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# [cite_start]Copie et installation des dépendances [cite: 2]
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# [cite_start]Copie tout le contenu du dossier data_pipeline/ vers /app [cite: 2]
COPY . .

# [cite_start]Configuration pour que Python trouve les scripts dans src/ [cite: 2]
ENV PYTHONPATH=/app

# [cite_start]Point d'entrée à la racine du service [cite: 2]
CMD ["python", "run_pipeline.py"]
