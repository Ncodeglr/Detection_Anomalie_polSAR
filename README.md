# Detection Anomalie polSAR (CVNN)

![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Ce projet propose un pipeline complet et modulaire pour la **détection d'anomalies non-supervisée** dans des images radar à synthèse d'ouverture polarimétrique (PolSAR). 

La particularité de ce projet réside dans son traitement natif des données à valeurs complexes grâce à l'utilisation de **réseaux de neurones à valeurs complexes (CVNN)**. En préservant simultanément l'amplitude et la phase du signal radar, le modèle capte des informations physiques cruciales. Le projet compare cette approche avec des méthodes de références (baselines) statistiques classiques et de Machine Learning.

## ✨ Fonctionnalités Principales

- **Deep Learning Complexe (CVNN)** : Modèle `LatentAutoEncoder` complexe pour la reconstruction d'images et la création d'un espace latent représentatif.
- **Détection Out-of-Distribution (OoD)** : Évaluation des anomalies par distance de Mahalanobis et erreurs de reconstruction.
- **Décompositions Physiques** : Visualisations et métriques basées sur les décompositions de Pauli, Krogager, H-Alpha et Cameron.
- **Génération d'Anomalies Physiques** : Injection synthétique d'erreurs d'antennes telles que la Diaphonie (*Crosstalk*) et le Déséquilibre de Gain (*Channel Gain Imbalance*).
- **Pipelines Multiples** : Comparaison entre méthodes Classiques (Stahel-Donoho, Cohérence, Entropie), Machine Learning (SVDD) et Deep Learning.

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

```text
Detection_Anomalie_polSAR/
├── configs/                     # Fichiers YAML (découpage azimutal, patchs, hyperparamètres)
├── cvnn/                        # Cœur de la librairie (Modèles complexes, Dataloaders PolSF/ALOS)
├── src/
│   ├── methode_Classique/       # Baseline stat/physique (Stahel-Donoho, Entropie, etc.)
│   ├── methode_MachineLearning/ # Baseline ML (ex: SVDD)
│   └── methode_DeepLearning/    # Pipeline auto-encodeur CVNN et OoD Detector
├── check_data.py                # Outil de visualisation du découpage des zones radar
└── README.md
```

## ⚙️ Exécution des pipelines

Le workflow de validation sépare rigoureusement les données :
- **Zone 1** : Apprentissage de la normalité (Train/Valid/Test).
- **Zone 2.1** : Test de référence sur des données pures (vérification de la Fausse Alarme - PFA).
- **Zone 2.2** : Test de robustesse avec injection de défauts physiques (Crosstalk, Gain).

Vous pouvez vérifier la découpe spatiale de vos données avec :
```bash
python3 check_data.py configs/config.yaml
```

### 1. Pipeline Deep Learning (Auto-encodeur Complexe)

Ce pipeline utilise un réseau de neurones pour modéliser la distribution saine et détecter les déviations.

```bash
# a. Entraînement de l'auto-encodeur sur la Zone 1
cd src/methode_DeepLearning/
python3 train_autoencoder.py ../../configs/config.yaml

# b. Évaluation des métriques de reconstruction (SSIM, PSNR, Cameron, H-Alpha)
python3 metrics_zone1.py ../../configs/config.yaml

# c. Évaluation de la détection d'anomalies OoD (Zone 2.1 et 2.2)
python3 eval_ood.py ../../configs/config.yaml
```
Les résultats, métriques (`.json`) et visualisations (reconstructions complètes, espace latent) sont sauvegardés dans le dossier `ml_results/`.

### 2. Pipeline Classique (Méthodes Physiques & Statistiques)

Méthodes de références se basant sur les matrices de covariance, la cohérence globale et l'entropie spectrale.

```bash
# a. Étalonnage du comportement normal (Stahel-Donoho) sur la Zone 1
cd src/methode_Classique/
python3 H0.py

# b. Extraction des tests et injection d'anomalies sur les Zones 2.1 et 2.2
python3 H1.py

# c. Évaluation et visualisation des métriques
python3 plot_metrics.py
```

### 3. Pipeline Machine Learning (SVDD)

Utilisation des caractéristiques extraites (Classique) pour entraîner un modèle classique de détection d'anomalie.

```bash
cd ../methode_MachineLearning/
python3 train_svdd.py
```