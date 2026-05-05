# Detection Anomalie polSAR (CVNN)

Ce projet implémente des réseaux de neurones à valeurs complexes (CVNN) pour la détection d'anomalies dans des images radar à synthèse d'ouverture polarimétrique (PolSAR).

## Prérequis

- Python 3.9+ (3.12 recommandé)
- PyTorch
- [Poetry](https://python-poetry.org/) (recommandé pour la gestion des dépendances) ou un environnement virtuel classique (`venv`).

## Installation

1. Cloner le dépôt et se rendre dans le dossier du projet.
2. Créer et activer un environnement virtuel :
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   # ou via poetry : poetry install
   ```