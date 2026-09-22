# Modèles locaux de prétri

Ce dossier contient uniquement les contrats et les licences. Les poids ne sont
pas suivis dans Git. Les licences Apache-2.0 de Grounding DINO et MIT de YuNet
et de VGG sont conservées avec leurs sommes de contrôle.

Le programme n’infère qu’avec des fichiers locaux vérifiés par
`manifest.json` :

- Grounding DINO tiny (`Apache-2.0`) pour proposer une zone de sujet guidée
  par le type de séance. La danse fait deux passes séparées : « a couple
  dancing together. », puis « a person dancing. » seulement si la première
  est vide ;
- MediaPipe Pose Landmarker lite (`Apache-2.0`) pour lier un visage au buste
  du sujet. Un visage n’est principal que si au moins la moitié de son aire
  est dans la boîte et que le nez d’un buste (au moins deux points parmi
  épaules et hanches dans la boîte d’origine) est à moins de 1,5 hauteur de
  visage. Les personnes en concurrence restent inconnues. Sans ce poids, il
  n’y a pas de repli par recouvrement ;
- le détecteur YuNet (`MIT`) dans `models/face` pour les visages et le repli
  de seuil ;
- la sortie VGG16 (`MIT`) comme signal facultatif, pas comme rejet.

Le téléchargement initial est volontaire. Il doit être effectué avec
`scripts/fetch-culling-models.sh`, qui vérifie la taille et le SHA-256 de DINO,
de YuNet et de Pose Landmarker lite. Le poids VGG issu de Google Drive n’est activé qu’avec une somme
SHA-256 fournie séparément par l’utilisateur ; sans cette preuve, le résultat
reste `unknown`.

Le score VGG natif est la moyenne de `sigmoid(fusion)` après padding
`reflect` aux multiples de 32, normalisation ImageNet, et recoupe hors
padding. Sur les poids autorisés, un flou gaussien a fait baisser cette
moyenne sur les 20 crops nets du diagnostic (delta médian −0,325) : une
moyenne plus basse est plus de flou, pas l’inverse. Les lots annotés se
recouvrent encore (nets jusqu’à 0,0249, mous jusqu’à 0,2981). Le point
historique `native_mean < 0,060` n’est pas adopté. Le score peut être
affiché ; il ne déclenche pas `focusReview`. L’étalonnage reste indisponible.

Le worker Python doit être installé dans un environnement utilisateur isolé
avec `torch`, `transformers`, `numpy`, `Pillow` et `mediapipe` (par exemple
un venv privé), puis sélectionné avec
`PICPORTAL_CULLING_PYTHON=/chemin/vers/venv/bin/python`.
PicPortal ne crée pas cet environnement et ne télécharge aucune dépendance à
l’ouverture d’un dossier. `transformers` est utilisé en mode
`local_files_only=True`.

Après installation, le worker `scripts/culling_worker.py` est exécuté
uniquement en local. Il n’envoie ni image ni chemin à un service distant.
L’absence du worker, de Python ou d’un poids valide ne bloque pas le moteur
historique : elle produit des signaux d’attribution, d’yeux et de focus
explicitement inconnus.
