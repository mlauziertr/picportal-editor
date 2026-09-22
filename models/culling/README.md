# Modèles locaux de prétri

Ce dossier contient uniquement les contrats et les licences. Les poids ne sont
pas suivis dans Git. Les licences Apache-2.0 de Grounding DINO et MIT de YuNet
et de VGG sont conservées avec leurs sommes de contrôle.

Le programme n’infère qu’avec des fichiers locaux vérifiés par
`manifest.json` :

- Grounding DINO tiny (`Apache-2.0`) pour proposer une zone de sujet guidée
  par le type de séance ;
- le détecteur YuNet (`MIT`) dans `models/face` pour les visages et le repli
  de seuil ;
- la carte de défocalisation VGG16 (`MIT`) comme signal de revue facultatif.

Le téléchargement initial est volontaire. Il doit être effectué avec
`scripts/fetch-culling-models.sh`, qui vérifie la taille et le SHA-256 de DINO
et de YuNet. Le poids VGG issu de Google Drive n’est activé qu’avec une somme
SHA-256 fournie séparément par l’utilisateur ; sans cette preuve, le résultat
reste `unknown`.

Le worker Python doit être installé dans un environnement utilisateur isolé
avec `torch`, `transformers`, `numpy` et `Pillow` (par exemple un venv privé),
puis sélectionné avec `PICPORTAL_CULLING_PYTHON=/chemin/vers/venv/bin/python`.
PicPortal ne crée pas cet environnement et ne télécharge aucune dépendance à
l’ouverture d’un dossier. `transformers` est utilisé en mode
`local_files_only=True`.

Après installation, le worker `scripts/culling_worker.py` est exécuté
uniquement en local. Il n’envoie ni image ni chemin à un service distant.
L’absence du worker, de Python ou d’un poids valide ne bloque pas le moteur
historique : elle produit des signaux d’attribution, d’yeux et de focus
explicitement inconnus.
