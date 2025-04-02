# PFE – Résolution d'un problème de plus court chemin via une BlackBox

Ce dépôt contient l'implémentation d'un ensemble de méthodes visant à résoudre un problème de plus court chemin dont les coûts d'arêtes sont générés dynamiquement par un module externe (`blackBox.exe`).

## Contenu du dépôt

```
Rendu_PFE/
├─ baseline.py            <- Méthode témoin (baseline) 
├─ blackBox.exe           <- Module externe
├─ blackbox_interface.py  <- Fonctions pour interagir avec blackBox.exe 
├─ execution_results_test.txt <- Exemple de résultats d’exécution
├─ graph_utils.py         <- Fonctions utilitaires pour générer ou manipuler un graphe
├─ main.py                <- Script principal (exemple d’exécution)
├─ monte_carlo.py         <- Méthodes Monte Carlo (classique, exploration, nested rollouts)
├─ q_learning.py          <- Approche Q-Learning 
└─ README.md              <- Vous êtes ici
```

## Pré-requis

- **Python 3** (>= 3.7)
- **Modules Python** : `numpy`, `networkx`, `tqdm` (installez-les via `pip install nom_du_module`)
- **blackBox.exe** : Binaire Windows. Pour un autre OS, un conteneur ou un environnement émulé peut être nécessaire.

## Utilisation

1. **Vérifier/Renseigner le chemin de `blackBox.exe`**  
   Dans le fichier `main.py`, adaptez éventuellement la variable/path qui fait référence à `blackBox.exe` si vous l’avez placé ailleurs.

2. **Exécuter le script principal**  
   ```bash
   python main.py
   ```
   - Celui-ci génère plusieurs graphe avec différents paramètres ( temps, densité, nombre de noeuds ), puis appelle différentes méthodes (baseline, Monte Carlo, Q-Learning) pour évaluer leurs performances.

3. **Analyser les résultats**  
   - Le fichier `execution_results_test.txt` peut contenir un exemple de trace ou de résultats.
   - Des courbes de convergences sont générés sur chaque ensemble de paramètres différent. 

## Méthodes Implémentées

- **Baseline** : Génère un chemin témoin (aléatoire, sans apprentissage).  
- **Monte Carlo** :  
  - Classique : teste de nombreux chemins aléatoires et conserve le meilleur.  
  - Avec Exploration : pénalisation des chemins coûteux via une mémoire de visites.  
  - Nested Rollouts : stratégie d’exploration imbriquée.  
- **Q-Learning** : Algorithme d’apprentissage par renforcement, met à jour une table `Q` en fonction des coûts retournés par `blackBox.exe`.

## Remarques

- Les **coûts** des arêtes sont retournés par `blackBox.exe` :  
  ```bash
  blackBox.exe <seed> <liste_id_liens,...>
  ```
  Chaque lien est identifié par un entier, et la **graine (`seed`) doit rester fixe** pour assurer la cohérence des tests.

- Les **hyperparamètres** (durée d’exécution, taux d’apprentissage, etc.) sont ajustables dans les fonctions correspondantes (voir `monte_carlo.py` ou `q_learning.py`).

## Auteurs

- **Abdel-Malek EMZIANE**  
- **Loucas TERCHANI**

*(Projet de fin d'études réalisé en partenariat avec CY Cergy Paris Université et Huawei Technologies France, supervisé par Sebastien Martin et Youcef Magnouche)*
