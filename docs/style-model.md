# Modèle de style v1 — « Appliquer mon style »

Un petit modèle local propose les 10 réglages de base dans le style du photographe, à partir de
l'image non retouchée. Il est entraîné en Python (`scripts/style/`), exporté en ONNX et exécuté
par l'éditeur via `ort` (`src-tauri/src/style_model.rs`). Aucun service cloud, aucun poids
pré-entraîné téléchargé.

## Cible

Clés du type `Adjustments` (`src/utils/adjustments.ts`), dans les unités des curseurs :

| Clé                                                     | Plage      | Arrondi |
| ------------------------------------------------------- | ---------- | ------- |
| `exposure`                                              | −5 … 5     | 0,01    |
| `contrast`, `highlights`, `shadows`, `whites`, `blacks` | −100 … 100 | 1       |
| `temperature`, `tint`, `vibrance`, `saturation`         | −100 … 100 | 1       |

Le modèle sort des valeurs brutes ; l'éditeur borne et arrondit comme les curseurs.
Hors v1 : courbes, HSL, étalonnage, masques, preset.

## Corpus

Un dossier (ou plusieurs, `--corpus` répétable), parcouru récursivement. Une photo compte si :

1. son sidecar `<photo>.<ext>.rrdata` a au moins un des 10 réglages différent de 0 ; sinon
2. son XMP `<photo>.xmp` contient des réglages de développement Lightroom (`crs:`).

Conversion Lightroom : même formules que `preset_converter.rs` (ombres × 1,5, température en
mireds relative à `AsShotTemperature`, teinte / 1,5). La fixture
`models/style/fixtures/lightroom-basic.xmp` est vérifiée par les tests Rust et Python.
Le Python lit aussi la forme élément (`<crs:Exposure2012>…`), que le convertisseur Rust ignore.

## Caractéristiques (`style-features-v1`, 55 valeurs)

Calculées sur une source sRGB « affichable » : le fichier décodé (JPEG, PNG, TIFF, WebP), ou
**le plus grand aperçu JPEG intégré** pour un RAW (même parcours d'IFD que
`largest_tiff_jpeg_preview`). L'orientation EXIF est appliquée.

1. Aperçu 64 × 64 : moyenne de blocs aux bornes entières `[⌊i·L/64⌋, ⌊(i+1)·L/64⌋)`.
2. Luminance `Y = 0,2126 R + 0,7152 G + 0,0722 B` (valeurs sRGB encodées).
3. Globales : moyennes R, G, B, Y ; écart-type Y ; percentiles de Y 1, 5, 10, 25, 50, 75, 90,
   95, 99 (rang le plus proche) ; parts écrêtées (Y ≥ 0,98, Y ≤ 0,02) ; chaleur `R − B` ;
   axe vert–magenta `G − (R + B)/2` ; saturation `max − min`.
4. Grille 4 × 4 : Y moyen et chaleur moyenne par case (32 valeurs).
5. EXIF : présence (ISO, temps de pose, focale tous lus), `log2(ISO/100)`, `log2(temps)`,
   `log2(focale)` ; valeurs neutres si absentes (100, 1/125 s, 50 mm).

Pas d'embedding visuel en v1 : aucun modèle ONNX libre < 30 Mo n'est livré dans le dépôt, et en
télécharger un sortait du cadre « sans téléchargement de poids ». Les caractéristiques restent
extensibles (changer `FEATURE_VERSION`).

La parité Python ↔ Rust est testée sur trois fixtures (`parity.png`, `parity.jpg`,
`parity-preview.dng`) : écart ≤ 1e-5 en PNG, ≤ 4e-3 en JPEG (décodeurs libjpeg vs zune-jpeg).

## Modèles et sélection

- **≥ 50 photos** : candidats ridge (α ∈ {0,1 ; 1 ; 10 ; 100}), kNN (k ∈ {3, 5, 8, 12}), MLP une
  couche tanh (16 ou 32 neurones, décroissance 1e-3 ou 1e-2, Adam plein lot, 800 époques),
  tous sur caractéristiques et cibles standardisées. 20 % des photos (≥ 10) sont mises de côté ;
  le candidat est choisi par validation croisée 5 plis sur le reste, évalué sur le jeu mis de
  côté, puis réentraîné sur tout le corpus pour l'export.
- **< 50 photos (repli)** : k-means (k = n/10 borné à 1…5) et **moyenne des réglages par
  grappe** ; évaluation par validation croisée 5 plis ; `manifest.fallback = true` et message
  explicite dans l'interface.

Tous les graphes ONNX ont la même interface : `features` float32 [N, 55] → `adjustments`
float32 [N, 10] ; standardisation incluse. Opset 17, IR 8 (compatible ONNX Runtime 1.22 utilisé
par `ort` 2.0.0-rc.10).

## Métriques

- MAE par réglage du modèle et de la **baseline « réglages moyens »** (moyenne du corpus
  d'entraînement), sur le jeu mis de côté.
- `relativeMae` = moyenne sur les réglages de MAE modèle / MAE baseline ; < 1 bat la baseline.
- Corpus synthétique uniquement : `presetRecovery`, part des photos dont la prédiction (réglages
  purement stylistiques) est la plus proche du preset qui l'a produite.

## Artefacts

`python3 -m scripts.style.train --corpus <dir> --out <dossier>` écrit :

- `style-model.onnx` ;
- `manifest.json` : `version`, `modelId` (`style-v1-<type>-<12 premiers hex du SHA-256>`),
  `featureVersion`, `featureCount`, `featureNames`, `outputKeys`, `outputRanges`, `kind`,
  `fallback`, `trainingSamples`, `minimumForLearnedModel`, `corpusFingerprint` (SHA-256 des
  matrices, sans chemin), `createdAt`, `seed`, versions de l'outillage, résumé de l'évaluation,
  `artifacts[{role, filename, sha256, bytes}]` ;
- `report.json` : métriques complètes, candidats, résumé du corpus (comptes, sans chemin).

L'éditeur refuse un modèle dont le contrat (version, caractéristiques, clés) diffère ou dont le
SHA-256 / la taille ne correspondent pas au manifeste. Même corpus + même graine ⇒ même fichier
ONNX (vérifié : SHA-256 identique sur deux entraînements).

Emplacements lus par l'éditeur, dans l'ordre :

1. `<données de l'application>/models/style/` (modèle personnel, application installée ;
   Linux `~/.local/share/com.getpicportal.PicPortalEditor/models/style`, macOS
   `~/Library/Application Support/com.getpicportal.PicPortalEditor/models/style`) ;
2. ressource `models/style` si un modèle est livré avec l'application (pas le cas en v1) ;
3. `models/style/` du dépôt (développement, sortie par défaut de `train.py`, ignoré par git).

Le modèle est rechargé automatiquement quand `manifest.json` change.

## Intégration dans l'éditeur

- **Panneau de retouche** : bouton `Palette` après l'ajustement auto, et entrée du sous-menu
  « Productivité » du menu contextuel de l'éditeur. Commande `calculate_style_adjustments` : ne
  touche pas le sidecar, renvoie un `patch` appliqué en une seule modification annulable
  (Ctrl/Cmd+Z). Les 10 réglages sont écrasés (geste explicite, spec UX MAX-19 §2.3).
- **Bibliothèque** : entrée « Appliquer mon style à N photos » sous l'ajustement auto.
  Commande `apply_style_to_paths(paths, skipEdited = true)`, qui passe par le même chemin que
  `apply_auto_adjustments_to_paths` (verrou de sidecar, synchronisation XMP, vignettes), mis en
  commun dans `update_adjustments_for_paths`. Les photos ayant un réglage de base **manuel**
  sont ignorées et comptées.
- **Provenance** : `adjustments.styleProvenance = { modelId, appliedAt, keys, values, previous }`.
  Un réglage est « au style » s'il vaut 0 ou exactement la valeur écrite par le style ; sinon il
  est manuel. `previous` garde les valeurs d'avant la première application (base d'un futur
  « Retirer mon style » / annulation de lot).
- **Erreurs** : codes `STYLE_…` mappés vers des textes i18n (`style.*`, EN et FR) ; aucune chaîne
  brute affichée.

## Commandes

```bash
python3 -m pip install -r scripts/style/requirements.txt   # numpy, pillow, onnx (+ onnxruntime pour les tests)
python3 -m scripts.style.train --corpus ~/Photos/Retouchées --out models/style
python3 -m scripts.style.extract --corpus <dir> --out dataset.npz        # extraction seule
python3 -m scripts.style.train --dataset dataset.npz --out models/style  # entraînement seul
python3 -m scripts.style.synth --out /tmp/style-synth --count 400        # corpus synthétique
python3 -m scripts.style.make_fixtures                                    # régénère les fixtures de parité
python3 -m unittest discover -s scripts/style/tests -t .
cd src-tauri && cargo test --lib style_model
node --test --import ./tests/resolveTypeScriptExtensions.mjs tests/styleModel.test.ts
```

## Corpus synthétique

`scripts/style/synth.py` rend des scènes procédurales (paysage, portrait, nuit, intérieur ;
aucune image téléchargée), les « prend » avec une erreur d'exposition (±1,2 EV) et une dominante
(chaud/froid, vert/magenta), et écrit comme retouche le preset connu de la scène
(`scripts/style/fixtures/synthetic-presets.json`) plus la correction de ces erreurs et un bruit
« humain ». Moitié `.rrdata`, moitié XMP Lightroom, 5 % non retouchées.

## Limites connues

- Le style appris est relatif à la source vue : l'aperçu JPEG intégré au RAW (rendu boîtier), pas
  le développement RAW de l'éditeur. Si le rendu de base diffère fortement, l'exposition proposée
  peut être biaisée ; à mesurer sur le corpus réel.
- EXIF des RAW : lu dans les conteneurs TIFF (NEF, ARW, CR2, DNG…). CR3, RAF, ORF, RW2 : EXIF
  absent pour les caractéristiques (drapeau à 0), aperçu trouvé par balayage des flux JPEG côté
  Python et par `rawler` côté Rust — parité non vérifiée sur ces formats.
- Sources 16 bits (PNG/TIFF) : Pillow les réduit à 8 bits ; parité non vérifiée.
- Balance des blancs : sans référence neutre, la dominante reste en partie confondue avec la
  couleur de la scène ; sur le corpus synthétique, la teinte bat à peine la baseline.
- Un réglage manuel remis exactement à 0 est considéré comme « non retouché ».
- Pas d'entraînement depuis l'interface en v1 (ligne de commande) ; la modale « Entraîner mon
  style », les badges « Style » par curseur, « Retirer mon style », l'annulation de lot et le
  raccourci Maj+S de la spec UX (MAX-19 §2) restent à faire.
