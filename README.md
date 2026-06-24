# Detection Anomalie polSAR (CVNN)

![Python Version](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Ce projet propose un pipeline complet et modulaire pour la **détection d'anomalies non-supervisée** dans des images radar à synthèse d'ouverture polarimétrique (PolSAR). 

La particularité de ce projet réside dans son traitement natif des données à valeurs complexes grâce à la bibliothèque **Complex-Valued Neural Networks (CVNN)**. En préservant simultanément l'amplitude et la phase du signal radar, le modèle capte des informations physiques cruciales. Le projet compare cette approche Deep Learning avec des méthodes de références (baselines) statistiques et de Machine Learning pour une évaluation complète.

## ✨ Fonctionnalités Principales

- **Deep Learning Complexe (CVNN)** : Modèle `UNet` complexe pour la modélisation de la distribution saine et la création d'un espace latent sémantique.
- **Détection Out-of-Distribution (OoD)** : Un détecteur d'anomalies hybride combinant la distance de Mahalanobis sur un espace latent **sémantique** (issu du UNet) et un espace **physique** (basé sur la décomposition de Pauli).
- **Génération d'Anomalies Physiques** : Injection synthétique d'erreurs d'antennes telles que la Diaphonie (*Crosstalk*) et le Déséquilibre de Gain (*Channel Gain Imbalance*).
- **Pipelines de Comparaison** : Évaluation rigoureuse et synchronisée de trois approches :
  - **Classique** : Basée sur la profondeur statistique de Stahel-Donoho.
  - **Machine Learning** : Support Vector Data Description (SVDD) sur des caractéristiques extraites.
  - **Deep Learning** : Détecteur OoD hybride (Sémantique + Physique).

## Prérequis

- Python 3.9+ (3.12 recommandé)
- PyTorch
- Poetry (recommandé) ou un environnement virtuel classique (`venv`).

## 🚀 Installation

1. Cloner le dépôt et se rendre dans le dossier du projet.
2. Créer et activer un environnement virtuel :
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Installer les dépendances (via pip ou poetry) :
   ```bash
   pip install -r requirements.txt
   # ou : poetry install
   ```

## 📁 Structure du projet

Le projet est organisé en modules distincts pour chaque méthode, garantissant une séparation claire des logiques tout en partageant une base de configuration commune.

```
Detection_Anomalie_polSAR/
├── configs/
│   └── config_Unet.yaml         # Fichier de configuration UNIQUE pour toutes les expériences
├── cvnn/                        # Sous-module Git: Librairie CVNN pour les opérations complexes
└── src/
    ├── methode_Classique/       # Baseline stat/physique (Stahel-Donoho)
    ├── methode_MachineLearning/ # Baseline ML (SVDD)
    ├── methode_DeepLearning/    # Pipeline UNet CVNN et Détecteur OoD
    └── shared_setup.py          # Script centralisé pour l'initialisation des données
```

## ⚙️ Workflow Expérimental Unifié

Pour garantir une comparaison juste et reproductible, toutes les méthodes s'appuient sur le même fichier `config_Unet.yaml` et sur une séparation géographique stricte des données :
- **Zone 1 (PolSF)** : Apprentissage de la distribution "saine" (Entraînement du UNet, calibration du SVDD et de Stahel-Donoho).
- **Zone 2.1** : Zone saine non-vue, utilisée pour mesurer le taux de fausses alarmes (PFA).
- **Zone 2.2** : Zone saine non-vue, utilisée pour l'injection d'anomalies et la mesure du taux de détection.

Le workflow se déroule en 3 étapes séquentielles.

### Workflow Principal : Basé sur UNet

### Étape 1 : Entraînement du modèle Deep Learning

Le modèle UNet doit être entraîné en premier, car ses poids sont utilisés par le pipeline d'évaluation Deep Learning.

```bash
# Se placer dans le dossier de la méthode Deep Learning
cd src/methode_DeepLearning

# Lancer l'entraînement du UNet sur la Zone 1
python3 train_unet.py ../../configs/config_Unet.yaml
```
Les poids du meilleur modèle sont sauvegardés dans `src/methode_DeepLearning/Unet_results/`.

### Étape 2 : Lancement des pipelines d'évaluation

Une fois le UNet entraîné, vous pouvez lancer les trois pipelines d'évaluation. Ils peuvent être lancés indépendamment, mais la méthode Machine Learning dépend des fichiers générés par la méthode Classique.

1.  **Méthode Classique (Stahel-Donoho)**
    Ce script calibre le modèle statistique et génère les caractéristiques (`features`) pour la méthode SVDD.
    ```bash
    cd src/methode_Classique
    python3 eval_H0.py   # Calibre le modèle sur la Zone 1
    python3 eval_H1.py   # Teste sur les Zones 2.1/2.2 et génère les .npy
    ```

2.  **Méthode Machine Learning (SVDD)**
    Ce script charge les caractéristiques extraites à l'étape précédente pour entraîner et évaluer le SVDD.
    ```bash
    cd src/methode_MachineLearning
    python3 train_svdd.py
    ```

3.  **Méthode Deep Learning (Détecteur OoD)**
    Ce script charge les poids du UNet pré-entraîné et évalue le détecteur hybride.
    ```bash
    cd src/methode_DeepLearning
    python3 eval_ood.py
    ```

### Étape 3 : Analyse des résultats

Chaque pipeline produit une sortie console standardisée pour une comparaison facile des taux de détection et des scores AUC-ROC.
- **Résultats Classiques** : Les scores sont dans `src/methode_Classique/test_results/`. Visualisez-les avec :
  ```bash
  cd src/methode_Classique && python3 plot_metrics.py
  ```
- **Résultats SVDD** : Les graphiques et scores sont dans `src/methode_MachineLearning/ml_results/`.
- **Résultats Deep Learning** : Les métriques (`.json`) et visualisations (espace latent) sont dans le dernier dossier de `src/methode_DeepLearning/Unet_results/`.