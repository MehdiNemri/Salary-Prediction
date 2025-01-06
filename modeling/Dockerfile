# Utiliser une image Python comme base
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier le script spécifique au conteneur
COPY modeling.py /app/

# Copier les fichiers généraux depuis la racine
COPY ../logging_config.py /app/
COPY ../requirements.txt /app/

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Définir la commande par défaut
CMD ["python", "modeling.py"]

