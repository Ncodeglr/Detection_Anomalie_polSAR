# Detection Anomalie polSAR (CVNN)

![Python Version](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-ee4c2c?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Ce projet propose un pipeline modulaire pour la **détection d'anomalies non-supervisée** dans des images radar à synthèse d'ouverture polarimétrique (PolSAR).

La particularité du projet réside dans son traitement natif des données à valeurs complexes grâce à la bibliothèque **Complex-Valued Neural Networks (CVNN)**. En préservant simultanément l'amplitude et la phase du signal radar, le modèle capte des informations physiques cruciales. Le projet compare l'approche Deep Learning avec une méthode de référence (baseline) statistique et physique.

Les anomalies sont des **erreurs de calibration d'antenne synthétiques** (diaphonie / déséquilibre de gain) injectées dans des données saines.

## ✨ Fonctionnalités principales

- **Deep Learning complexe (CVNN)** : Autoencodeur complexe (`AutoEncoder`) pour la modélisation de la distribution saine et la création d'un espace latent sémantique.
- **Détection Out-of-Distribution (OoD)** : Détecteur d'anomalies **hybride** combinant la distance de Mahalanobis sur :
  - un espace latent **sémantique** (issu de l'autoencodeur, réduit par PCA) ;
  - un espace **physique** (matrice de Gram des vecteurs de rétrodiffusion projetée dans l'espace tangent log-euclidien de la variété SPD).
  - Ajustement robuste de la covariance en 2 passes (rejet des ~5 % de diffuseurs urbains extrêmes) et seuils calibrés à une PFA cible.
- **Génération d'anomalies physiques** : Injection synthétique d'erreurs d'antenne — diaphonie (*Crosstalk*, `C' = D · C · Dᴴ` avec `D = kron(D₂, D₂)`, `D₂ = [[1, δ], [δ, 1]]`) et déséquilibre de gain de canal.
- **Méthode Classique** : Profondeur statistique de Stahel-Donoho (approximation par projections aléatoires) sur 16 caractéristiques polarimétriques extraites par patch.

## Prérequis

- Python 3.10+
- PyTorch 2.7
- Un environnement virtuel (`venv`) — voir l'installation.

## 🚀 Installation

1. Cloner le dépôt **avec ses sous-modules** :
   ```bash
   git clone --recurse-submodules <url_du_depot>
   cd Detection_Anomalie_polSAR
   # si déjà cloné sans --recurse-submodules :
   git submodule update --init
   ```
   Le dossier `cvnn/` est un sous-module Git (`https://github.com/Ncodeglr/cvnn.git`) qui fournit
   le chargement des données (`ALOSDataset`), les modèles complexes (`AutoEncoder`), le chargement
   de configuration et les utilitaires d'entraînement.

2. Créer et activer l'environnement virtuel :
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
   `cvnn` n'est pas installé comme paquet : chaque script d'entrée manipule `sys.path` pour
   l'importer depuis `cvnn/src`.

4. Placer les données brutes localement (non suivies par Git) :
   - `data/PolSF/` — scène PolSF (Zone 1).
   - `data/SAN_FRANCISCO_ALOS2/` — scène ALOS-2 San Francisco (fichiers CEOS `IMG-{HH,HV,VH,VV}-*`).

## 📁 Structure du projet

```
Detection_Anomalie_polSAR/
├── configs/
│   └── config_Unet.yaml            # Fichier de configuration UNIQUE pour toutes les expériences
├── cvnn/                           # Sous-module Git : librairie CVNN (opérations complexes)
└── src/
    ├── shared_setup.py             # Point de coordination : split géographique, loaders, générateur d'anomalies partagé
    ├── synthetic_parameter_generator.py  # Tirage des paramètres d'anomalie (amplitude ~ N, phase ~ VonMises)
    ├── visualize_zones.py          # Visualisation du découpage des zones
    ├── methode_Classique/          # Baseline statistique/physique (Stahel-Donoho)
    └── methode_DeepLearning/       # Autoencodeur complexe + détecteur OoD hybride
```

### `src/shared_setup.py`

Toutes les méthodes l'importent. Il **impose** le découpage géographique (`ZONES_CONFIG`) et
écrase les `crop_coordinates` du YAML, de sorte que le split est identique quelle que soit la
config :

- **Zone 1** (lignes 0–8000) : distribution saine d'apprentissage — entraîne l'autoencodeur,
  calibre Stahel-Donoho.
- **Zone 2.1** (lignes 8200–15000) : région saine non-vue — mesure du taux de fausses alarmes (PFA).
- **Zone 2.2** (lignes 15200–20000) : région saine non-vue, en 3 bandes — injection d'anomalies
  et taux de détection.

`get_shared_anomaly_generator()` renvoie le générateur de `δ` et sa graine, pour que chaque
méthode injecte **les mêmes** paramètres d'anomalie.

## ⚙️ Workflow expérimental unifié

Toutes les méthodes s'appuient sur le même `config_Unet.yaml` (`seed: 42`, `anomaly_seed: 1234`
par défaut) et sur la séparation géographique stricte ci-dessus.

> ⚠️ **Chaque script doit être exécuté depuis son propre dossier de méthode.** Les chemins de
> sortie sont relatifs au répertoire courant. Chaque exécution écrit un sous-dossier horodaté
> (`run_YYYYMMDD_HHMMSS/` ou nom de run wandb), et les étapes suivantes sélectionnent
> automatiquement le run **le plus récent** (par date de modification). Ne pas lancer depuis la
> racine du dépôt.

### Étape 1 : Entraînement du modèle Deep Learning

L'autoencodeur doit être entraîné en premier, car ses poids alimentent l'évaluation OoD.

```bash
cd src/methode_DeepLearning
python3 train_autoencoder.py ../../configs/config_Unet.yaml
```
Les poids du meilleur modèle sont sauvegardés dans
`src/methode_DeepLearning/DL_results/<run>/best_weights_autoencoder.pt`.

Le logging Weights & Biases est câblé dans le script (projet `polSAR-anomaly-detection-DL`).
Utiliser `WANDB_MODE=offline` pour désactiver la synchronisation.

### Étape 2 : Pipelines d'évaluation

Les deux pipelines sont indépendants.

1. **Méthode Classique (Stahel-Donoho)**
   ```bash
   cd src/methode_Classique
   python3 eval_H0.py       # calibre le modèle statistique sur la Zone 1  -> data_calibration/run_*/
   python3 eval_H1.py       # teste les Zones 2.1/2.2, injecte les anomalies -> test_results/run_*/
   python3 plot_metrics.py  # seuil PFA, ratios accept/reject, AUC-ROC, histogrammes
   ```

2. **Méthode Deep Learning (Détecteur OoD)**
   ```bash
   cd src/methode_DeepLearning
   python3 eval_ood.py      # charge les derniers poids de DL_results/ et évalue le détecteur hybride
   ```

Script optionnel : `metrics_zone1.py` (métriques de reconstruction / visualisation de l'espace
latent sur la Zone 1).

### Étape 3 : Analyse des résultats

- **Résultats Classiques** : scores dans `src/methode_Classique/test_results/` ; visualisation via
  `plot_metrics.py`.
- **Résultats Deep Learning** : métriques (`.json`) et visualisations dans le dernier dossier de
  `src/methode_DeepLearning/DL_results/`.

## 🗺️ Architecture des méthodes

### `src/methode_Classique/`

- `feature_extraction.py` : patch `(B,4,H,W)` complexe → covariance moyennée spatialement
  `C = k kᴴ / Np` → **16 caractéristiques** = 12 composantes de cohérence complexe
  (`γ_ij = C_ij/√(C_ii C_jj)`, parties réelle+imaginaire des 6 paires hors-diagonale) + 4
  intensités diagonales en dB.
- `H0.py` (`DepthCalibrator`) : approximation de l'outlyingness de Stahel-Donoho par projections
  aléatoires — M=8000 directions unitaires dans R¹⁶, médiane/MAD par direction,
  `O(x) = max_v |vᵀx − med| / MAD`, `score = 1/(1+O)` (élevé = normal). Paramètres sauvegardés
  (`V_directions.npy`, `Train_medians.npy`, `Train_mads.npy`).
- `H1.py` : modèle d'anomalie numpy sur la matrice de covariance 4×4.

### `src/methode_DeepLearning/`

- `train_autoencoder.py` : entraîne l'`AutoEncoder` complexe de CVNN (reconstruction) sur la Zone 1.
- `ood_detector.py` (`OOD_Detector`) : score de Mahalanobis hybride (sémantique PCA + physique
  log-euclidien), ajustement robuste de la covariance, seuils calibrés sur le split de validation
  de la Zone 1.
- `anomalies.py` : modèle d'anomalie `torch.nn.Module` (`Crosstalk`) opérant sur le vecteur de
  rétrodiffusion complexe brut, avant le réseau.

## Conventions & pièges

- Les dossiers de sortie sont **relatifs au CWD** et référencés par chemin relatif entre méthodes.
  Ne pas lancer depuis la racine.
- La sélection du « dernier run » est `max(..., key=os.path.getmtime)` partout : un `touch` sur un
  vieux dossier redirige silencieusement les étapes suivantes.
- `DepthCalibrator.generate_vectors()` n'est **pas** seedé : la matrice de projection `V` diffère
  d'un run H0 à l'autre malgré `seed: 42`.
- `config["data"]["dataset"]["base_dir"]` est un chemin absolu propre à une autre machine ; c'est
  `trainpath` (`./data/...`, résolu depuis la racine du dépôt) qui est réellement utilisé.
