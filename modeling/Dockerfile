# Utiliser une image Python comme base
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers spécifiques au conteneur
COPY logging_config.py /app/
COPY requirements.txt /app/
COPY modeling.py /app/

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Définir la commande par défaut
CMD ["python", "modeling.py"]
