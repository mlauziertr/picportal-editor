# Spécification UX v1 — tri assisté, « Appliquer mon style », publication PicPortal

Tâche : MAX-19 · Autrice : Inès (design produit) · 24 septembre 2026
Base analysée : `backup/picportal-editor-platform-integration-256-513173c5` (SHA `513173c5`).
Textes d’interface : `docs/ux/i18n/en.json` et `docs/ux/i18n/fr.json` (fragments à fusionner dans `src/i18n/locales/`).

## 0. Principes transverses

Ces règles valent pour les trois parcours. Chaque critère de réception y renvoie.

| # | Règle | Lentille |
|---|---|---|
| P1 | **Proposer, puis appliquer.** Une analyse (tri, style) ne modifie rien tant que l’utilisateur n’a pas cliqué sur une action d’application explicite. Chaque application en lot a son « Annuler ». | Nielsen 3 (contrôle), pardon |
| P2 | **Les décisions manuelles gagnent.** Une note, une étiquette ou un réglage posé à la main n’est jamais écrasé en silence. | Norman (contraintes), confiance |
| P3 | **Tout traitement long** (> 1 s) affiche : étape en clair, compteur `n sur N`, barre déterminée, bouton Annuler toujours visible (pas au survol). Feedback de clic < 100 ms. | Doherty, Nielsen 1 |
| P4 | **Aucune erreur brute.** Les chaînes Rust (`LOCAL_CULLING_…`, `PicPortal login failed: …`) ne s’affichent jamais telles quelles. Le backend renvoie un **code**, le front mappe vers un texte i18n ; le détail brut va derrière « Copier le détail technique ». | Nielsen 9 |
| P5 | **Étiquettes visibles.** Plus de champ dont le seul libellé est le placeholder. | WCAG 3.3.2, reconnaissance plutôt que rappel |
| P6 | **Couleur jamais seule.** Toute catégorie, provenance ou statut porte aussi un libellé ou une icône. | WCAG 1.4.1 |
| P7 | **Réutiliser le système existant.** Tokens Tailwind du thème (`bg-bg-primary`, `bg-surface`, `bg-card-active`, `text-text-primary`, `text-text-secondary`, `text-accent`, `border-border-color`), composants `Button`, `Switch`, `Text` (`TextVariants.title/heading/small/label`), `Dropdown`, `CollapsibleSection`, `ConfirmModal`, toasts `react-toastify`. Échelle d’espacement Tailwind existante (1, 2, 3, 4, 5, 6) ; aucune valeur arbitraire. | Jakob, cohérence |
| P8 | **Local d’abord, dit clairement.** Le tri et le style tournent sur la machine ; on le dit une fois, sans répétition anxiogène. | Confiance |

**Changements de système proposés (à valider par Chase)** — aucun nouveau token de couleur. Deux petits composants réutilisables :

- `ProgressBlock` (étape + compteur + barre + bouton Annuler) : remplace les trois implémentations ad hoc (modal de tri, bouton d’export, futur entraînement). Réutilisable par Denoise, HDR, Panorama.
- `StatusNote` (icône + texte, variantes `info | warning | error | success`, action facultative) : remplace les `div` jaunes et les `status` texte brut. Couleurs : `text-accent` (info), `yellow-500` (warning, déjà utilisé), `red-500` (error, déjà utilisé), `green-500` (success, déjà utilisé) — toujours avec icône lucide (`Info`, `AlertTriangle`, `XCircle`, `CheckCircle`).

---

## 1. Tri assisté (culling)

### 1.1 Existant et écarts

| Existant (`513173c5`) | Écart | Sévérité |
|---|---|---|
| `AutomaticCullingButton` dans l’en-tête bibliothèque, ouvre `CullingModal` | OK, on garde | — |
| `CullingModal` : options, puis spinner + barre pendant l’analyse | **Aucun bouton Annuler pendant l’analyse** ; pas de commande backend d’annulation | Bloquant |
| Étapes émises par Rust en anglais (`"Analyzing images locally..."`) | Non traduisibles | Majeur |
| À la fin, `onComplete` → `handleApplyCulling` **écrit immédiatement** notes et étiquettes | Contredit P1 : l’utilisateur découvre les résultats après coup, sans annulation | Bloquant |
| `CullingResultsPanel` : vignette = texte « Photo » / « 2 faces » | Impossible de juger une photo sans la voir | Bloquant |
| Catégories `selected, highlights, duplicate, blurred, closedEyes, unrated` | Similaires affichés à plat, pas par groupe | Majeur |
| Détail : `eyeState`, `subjectStatus` affichés en valeur brute (`unknown`, `open`) | Non traduits | Mineur |
| Worker sujet/pose absent → « unknown » découvert après l’analyse | Doit être annoncé **avant** le lancement | Majeur |
| `fr.json` : ~60 clés `modals.culling.*` encore en anglais | Corrigé par le fragment i18n | Majeur |
| Photos protégées (`getCullingProtectedPaths`) comptées après coup | Doivent être visibles pendant la revue | Majeur |

### 1.2 Parcours

```
Bibliothèque ──[Tri assisté]──► Options + état des modèles ──[Lancer l’analyse]──► Progression
                                                                                  │  [Annuler l’analyse] → toast « Analyse annulée… » → bibliothèque inchangée
                                                                                  ▼
                                                   Propositions de tri (revue par catégorie, ajustements)
                                                                                  │  [Fermer] → confirmation « Fermer sans appliquer ? »
                                                                                  ▼
                                                   [Appliquer les décisions à N photos] → toast « Décisions appliquées… [Annuler] »
```

### 1.3 Écran A — Options (`CullingModal`, état initial)

```
┌────────────────────────────────────────────────────────────────┐
│ ✦ Tri assisté                                              [×] │
│ /Photos/2026-09-12 Mariage                                     │
│                                                                │
│ ┌ 412 photos compatibles dans ce dossier ─────────────────────┐ │
│ │ Toutes les photos compatibles du dossier sont analysées…    │ │
│ │ Rien n’est modifié avant que vous appliquiez les décisions. │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                │
│ Analyse locale                        Les photos ne quittent   │
│  ✓ Netteté et photos similaires  Prêt   pas cet ordinateur.    │
│  ✓ Visages et yeux               Prêt                          │
│  ○ Sujet et pose                 Indisponible  ⓘ               │
│                                                                │
│ Nombre de photos à retenir                                     │
│ [ Très peu | Peu | ■Standard | Davantage ]                     │
│ Tolérance au flou                                              │
│ [ Tolérante | ■Modérée | Stricte ]                             │
│                                                                │
│ ▸ Options de détection                                         │
│                                        [Annuler] [Lancer l’analyse] │
└────────────────────────────────────────────────────────────────┘
```

- **Bloc « Analyse locale »** (nouveau) : alimenté par une commande de pré-vol `culling_capabilities` (à créer, voir §1.9) appelée à l’ouverture. Pendant l’appel : « Vérification des modèles locaux… » (≤ 400 ms attendu, sinon squelette de 3 lignes). Chaque ligne = icône (`CheckCircle` / `CircleDashed`) + libellé + statut texte. Un `ⓘ` ouvre le texte explicatif (`capabilities.workerMissing` ou `capabilities.facesMissing`).
- Si **visages/yeux indisponibles** : le `Switch` « Yeux fermés » dans Options de détection est désactivé, avec la raison en `TextVariants.small`. Le tri reste lançable.
- Si **sujet/pose indisponible** : `Switch` « Sujet et pose (facultatif) » désactivé, même logique. Défaut de ce switch : **désactivé** quand le worker est absent (aujourd’hui `detectSubject: true` par défaut).
- « Options de détection » reste replié (divulgation progressive). Le `select` de profil de sujet n’apparaît que si le switch Sujet est actif et disponible.
- Le `×` texte actuel est remplacé par l’icône `X` de lucide avec `aria-label` (cohérent avec `CullingResultsPanel`).
- **Dossier vide** : bouton « Lancer l’analyse » désactivé, bloc de portée en `StatusNote warning` avec `emptyFolder`.
- Focus initial sur « Lancer l’analyse » ; `Échap` = Annuler ; `Entrée` = Lancer.

### 1.4 Écran B — Progression

```
┌────────────────────────────────────────────────┐
│ ✦ Tri assisté                                  │
│                                                │
│ Analyse des photos                  128 sur 412 │
│ ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│ Vous pouvez annuler à tout moment.             │
│ Rien n’a encore été modifié.                   │
│                                  [Annuler l’analyse] │
└────────────────────────────────────────────────┘
```

- `ProgressBlock`. Étapes par **code** émis par Rust (`preparing | analyzing | subject | grouping`) → clés `modals.culling.stage.*`. Le spinner central de 56 px est supprimé (la barre suffit ; effet esthétique-utilisabilité, moins de bruit).
- Barre : `h-2 rounded-full bg-bg-primary`, remplissage `bg-accent`, `role="progressbar"` avec `aria-valuenow/max` et `aria-valuetext` = texte du compteur. Sous `prefers-reduced-motion`, pas de `transition-all`.
- **Annuler l’analyse** : bouton secondaire (style « ghost » existant `text-text-secondary hover:bg-bg-primary`). Au clic : libellé « Annulation… », bouton désactivé ; à la confirmation backend (événement `culling-cancelled`), fermeture de la modale + toast info `cancelled`. Pas de confirmation préalable : rien n’est perdu (P1).
- Fermer la fenêtre ou changer de dossier pendant l’analyse = annulation.
- Étape « Enregistrement… » (`savingResults`) supprimée de cette modale : l’enregistrement n’a lieu qu’à l’application (§1.6).

### 1.5 Écran C — Propositions de tri (`CullingResultsPanel`)

Surface existante (panneau plein écran sur la bibliothèque) conservée. Mise en page 1440×900 :

```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ Propositions de tri                                                              [X] │
│ /Photos/2026-09-12 Mariage · 412 photos analysées · rien n’est encore appliqué        │
│                                                                                       │
│ (●Retenues 96) (●Points forts 18) (●Similaires 143) (●Floues 37) (●Yeux fermés 12) (●À vérifier 106) │
├──────────────────────────────────────────────────────┬────────────────────────────────┤
│ Groupe 3 · 5 photos                                  │ IMG_0412.CR3                   │
│ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐                   │ [ aperçu 16:9 ]                │
│ │img │ │img │ │img │ │img │ │img │                   │ Note proposée 4/5 · Qualité 82%│
│ │★Meilleure│ │    │ │ 🔒 │ │    │                   │ Netteté · Visages 2 · Yeux Ouverts │
│ └────┘ └────┘ └────┘ └────┘ └────┘                   │ Pourquoi cette proposition     │
│ Groupe 4 · 3 photos                                  │  ✓ Très proche d’une autre…    │
│ …                                                    │ [Garder] [Rejeter] ★★★★☆       │
│                                                      │ Revenir à la proposition        │
├──────────────────────────────────────────────────────┴────────────────────────────────┤
│ 290 notes et 180 étiquettes seront écrites · 14 de vos décisions conservées           │
│                                              [Fermer] [Appliquer les décisions à 290 photos] │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

**Catégories** (ordre = ordre de travail : ce qu’on garde, puis ce qu’on écarte, puis l’incertain) :

| Clé | Libellé FR | Contenu | Correspondance brief |
|---|---|---|---|
| `selected` | Retenues | meilleures photos | — |
| `highlights` | Points forts | points forts techniques | — |
| `duplicate` | Similaires | **affichées par groupe** (`similarGroups`) : en-tête `groupLabel`, meilleure photo en premier avec badge « Meilleure du groupe » | similaires |
| `blurred` | Floues | | flous |
| `closedEyes` | Yeux fermés | | yeux fermés |
| `unrated` | À vérifier | aucun signal fiable, yeux ou sujet inconnus | inconnus |

- Pastilles : on garde le point coloré existant **et** le libellé + compteur (P6). Si une détection n’a pas tourné (désactivée ou modèle absent), la pastille affiche « Yeux fermés · Non analysé » en `text-text-secondary`, non cliquable, `aria-disabled`.
- **Vignettes réelles** : même source d’image que `LibraryGrid` (cache de vignettes de la bibliothèque). Carré `aspect-square rounded`, nom de fichier tronqué dessous, note proposée en étoiles. Pendant le chargement : squelette `bg-bg-primary animate-pulse` (désactivé en mouvement réduit).
- **Badge décision manuelle** : icône `Lock` 14 px en coin supérieur droit + tooltip `manualBadgeHint`. Ces photos restent visibles dans leur catégorie (pour comprendre le tri) mais sont exclues du compte d’application. Dans le détail : `StatusNote info` « Votre note est conservée ».
- **Ajustement par photo** (détail) : `Garder` / `Rejeter` (boutons segmentés) et étoiles 0–5 cliquables. Une photo modifiée affiche le badge texte « Modifié par vous » et un lien « Revenir à la proposition ». Raccourcis identiques à la bibliothèque : `0`–`5` note, flèches pour naviguer, `Échap` ferme le détail puis le panneau. « Rejeter » = sémantique existante `actionReject` (étiquette rouge, note 0) : **aucun fichier n’est supprimé ni déplacé**.
- **Détail** : les valeurs brutes (`eyeState`, `subjectStatus`, `qualityMethod`) passent par i18n (`eyes.open/closed/unknown`, `categoryNotAnalyzed`). Les méthodes techniques (`yunet;fallback-error…`) ne s’affichent pas ; elles vont dans « Copier le détail technique » si besoin. La ligne « Sujet » n’apparaît que si le module a tourné.
- **Catégorie vide** : `noResults` centré, `text-text-secondary`.
- Viewport étroit (< 1024 px, `lg:` actuel) : détail sous la grille, déjà géré par la grille existante ; la barre d’application reste collée en bas.

### 1.6 Application des décisions

- Barre inférieure fixe : `applySummary` (compteurs calculés en direct, ajustements inclus) + `Fermer` (secondaire) + `Appliquer les décisions à N photos` (primaire `Button`). N = photos dont la note ou l’étiquette changera réellement (hors protégées, hors inchangées). N = 0 → bouton désactivé.
- Clic : bouton « Application des décisions… » désactivé, puis écriture via `handleApplyCulling` existant (ordre notes → étiquettes conservé). Succès : fermeture du panneau, toast succès `applied` avec action **Annuler** (8 s, et le toast reste tant que le pointeur est dessus). Annuler restaure note/étiquette/flags `*_is_manual` **antérieurs** des chemins écrits, puis toast `undone`.
- Échec partiel : le panneau reste ouvert ; `StatusNote warning` en bas : `persistenceFailure` + `retryFailed` (réécrit seulement les échecs) + `showFailed` (liste des noms de fichiers).
- `Fermer` avec des propositions non appliquées → `ConfirmModal` : `discardConfirmTitle` / `discardConfirmBody` / action `discardConfirmAction` (bouton neutre, pas rouge : rien n’est détruit sur disque). Pas de confirmshaming.
- Le switch d’options « Proposer des notes et des étiquettes de couleur » (`autoAssignStars`) désactivé → le panneau sert de revue seule ; la barre d’application est masquée.

### 1.7 Erreurs

| Cas | Détection | Présentation | Action |
|---|---|---|---|
| Worker sujet/pose absent (`LOCAL_CULLING_WORKER_UNAVAILABLE`, `…_WORKER_START_FAILED`) | pré-vol | ligne « Sujet et pose · Indisponible » + ⓘ `capabilities.workerMissing` ; jamais bloquant | aucune |
| Modèles sujet/pose absents (`…_SUBJECT_MODEL_UNAVAILABLE`, `…_POSE_MODEL_UNAVAILABLE`) | pré-vol | idem | aucune |
| Modèles visages absents/altérés (`…_MODEL_MANIFEST_UNAVAILABLE`, `…_MANIFEST_MISMATCH`, `…_ARTIFACT_MISSING`, `…_LICENSE_*`) | pré-vol | ligne « Visages et yeux · Indisponible » + `capabilities.facesMissing` ; switch Yeux fermés désactivé ; pastille « Yeux fermés · Non analysé » | aucune (réinstallation = support) |
| Worker qui plante en cours d’analyse | pendant | l’analyse continue sans sujet ; en résultats, pastille/ligne « Non analysé » | aucune |
| Photo iCloud non téléchargée | pendant | comptée dans `failedAnalysisCount` ; nom listé via `showFailed` avec `errors.icloud` | `retryFailed` après téléchargement |
| Photo illisible | pendant | idem, texte générique | `retryFailed` |
| Échec global (panique, verrou empoisonné, etc.) | pendant | Écran d’erreur de la modale : `XCircle` + `cullingFailed` + `errors.generic` ; boutons `Fermer` / `Réessayer` (relance avec les mêmes options) + lien `copyDetails` | Réessayer |
| Analyse déjà en cours | lancement | toast `errors.alreadyRunning` | — |

### 1.8 Critères de réception — tri

1. Pendant l’analyse, « Annuler l’analyse » est visible sans survol ; après clic, aucune note/étiquette n’a changé (vérifier le sidecar d’une photo avant/après).
2. À la fin de l’analyse, **aucune écriture** n’a eu lieu tant que « Appliquer les décisions » n’est pas cliqué.
3. Chaque catégorie affiche des vignettes réelles ; « Similaires » est groupé avec la meilleure photo en tête.
4. Une photo notée à la main avant l’analyse porte le badge cadenas, n’est pas comptée dans N et garde sa note après application.
5. « Annuler » dans le toast restaure exactement les valeurs antérieures (note, étiquette, flags manuels).
6. Worker absent : le bloc « Analyse locale » le signale **avant** le lancement ; le tri aboutit ; aucune chaîne `LOCAL_CULLING_*` n’est visible.
7. Modèles visages absents : switch Yeux fermés désactivé avec raison ; pastille « Non analysé ».
8. Interface FR entièrement en français (aucune clé `modals.culling.*` identique à l’anglais, hors nombres et « Standard »/« Portrait »/« Style »).
9. Clavier seul : ouvrir, lancer, annuler, naviguer dans les catégories, appliquer. Focus visible partout.

### 1.9 Dépendances techniques (pour Théo, arbitrage Sam si API)

- `culling_capabilities` → `{ sharpness: ready, faces: ready|unavailable, subject: ready|unavailable, reasonCode }` (réutilise `select_model_directory` et la vérification de manifeste de `subject_inference.rs`).
- `cancel_culling` + drapeau d’annulation vérifié entre images ; événement `culling-cancelled`.
- `culling-progress` : ajouter `stageCode` (garder `stage` pour compatibilité).
- Découpler `culling-complete` de l’écriture : `CullingModal` ne doit plus appeler `onComplete → handleApplyCulling` ; l’écriture part du bouton d’application.
- Journal d’annulation : conserver `{path, rating, colorLabel, ratingIsManual, colorLabelIsManual}` antérieurs pour le lot écrit (mémoire suffit pour v1).

---

## 2. « Appliquer mon style »

Rien n’existe. Le modèle (MAX-18, Sam) prédit 10 réglages de base : exposition, contraste, hautes lumières, ombres, blancs, noirs, température, teinte, vibrance, saturation. La spec ci-dessous ne suppose rien d’autre.

### 2.1 Où vit l’action

| Contexte | Point d’entrée | Composant existant réutilisé |
|---|---|---|
| **Panneau de retouche** (une photo) | Bouton icône dans l’en-tête de `ControlsPanel`, **juste après** le bouton auto (`PencilSparkles`). Icône lucide `Palette`, tooltip `style.applyTooltip`. | même classe que les boutons voisins (`p-2 rounded-full hover:bg-surface`) |
| Panneau de retouche, menu contextuel | Sous-menu « Productivité », entrée « Appliquer mon style » juste sous « Réglage automatique » | `useAppContextMenus` (menu éditeur) |
| **Bibliothèque / lot** | Menu contextuel de vignette, sous-menu « Productivité », entrée `style.applyToSelection` (N) sous `autoAdjust` | `useAppContextMenus` (menu vignette) |
| Raccourci | nouvelle action `apply_my_style` dans la liste de `src/utils/keyboardUtils.ts` (section `editing`, donc reconfigurable), défaut `Maj+S` (`['shift','KeyS']`, libre ; `S` seul = recadrage) | liste de raccourcis existante |
| Entraînement et gestion | Réglages de l’application, nouvelle section « Mon style » ; et lien « Entraîner mon style… » depuis tout point d’entrée quand aucun style n’existe | page Réglages existante, `CollapsibleSection` |

Pas de bouton permanent dans l’en-tête bibliothèque : l’en-tête porte déjà le tri assisté ; un second bouton « IA » concurrent brouillerait la hiérarchie (Hick, Von Restorff).

**Sans style entraîné** : les entrées restent visibles (découvrabilité) mais ouvrent un popover (éditeur) ou la modale d’entraînement (bibliothèque) avec `style.noModel` + `style.noModelHint` + bouton `style.train`. Pas d’entrée grisée muette.

### 2.2 Entraînement (modale « Entraîner mon style »)

```
┌──────────────────────────────────────────────────────────────┐
│ Entraîner mon style                                      [X] │
│ Choisissez des dossiers de photos que vous avez retouchées…  │
│                                                              │
│ Dossiers                                                     │
│  /Photos/Mariages 2025                       [Retirer]       │
│  /Photos/Portraits                           [Retirer]       │
│  [+ Ajouter un dossier…]     ☑ Inclure les sous-dossiers     │
│                                                              │
│ ┌ 142 photos retouchées trouvées ───────────────────────────┐ │
│ │ 98 retouchées dans PicPortal Editor · 44 avec XMP         │ │
│ │ Lightroom · 17 ignorées ⓘ                                 │ │
│ │ ✓ Corpus de bonne taille.                                 │ │
│ └───────────────────────────────────────────────────────────┘ │
│                                   [Annuler] [Lancer l’entraînement] │
└──────────────────────────────────────────────────────────────┘
```

- **Choix du corpus** : un ou plusieurs dossiers (sélecteur de dossier natif Tauri). Le comptage se relance à chaque ajout/retrait (« Recherche des photos retouchées… », squelette). Une photo compte si elle a un `.rrdata` ou un XMP Lightroom avec au moins un des 10 réglages différent du défaut.
- **Seuils** (à confirmer par Sam avec le plan : repli kNN sous 50) :
  - `< 20` : `StatusNote error` `tooSmall` (min = 20), bouton désactivé.
  - `20–49` : `StatusNote warning` `small` (recommandé = 50), bouton actif.
  - `≥ 50` : `StatusNote success` `good`.
- **Progression** : même modale, `ProgressBlock` avec les étapes `reading → features (n sur N) → fitting → checking`, `localHint`, bouton « Annuler l’entraînement ». La modale peut être fermée : l’entraînement continue et un indicateur discret (icône `Palette` animée + %) apparaît dans la section Réglages « Mon style » ; fin signalée par toast. Un seul entraînement à la fois.
- **Annulation** : l’ancien style reste actif (`training.cancelled`). Aucun fichier de modèle partiel ne remplace l’ancien (écriture atomique).
- **Fin** : `training.done` + `doneBody` avec la fiabilité. Fiabilité = erreur moyenne sur un jeu mis de côté, ramenée à 3 niveaux (seuils définis par Sam) ; toujours un mot + une icône, jamais une couleur seule.
- **Section Réglages « Mon style »** : `status` (nombre, date au format local), fiabilité, boutons `retrain`, `delete` (→ `ConfirmModal` `deleteConfirm*`, bouton destructif rouge existant).

### 2.3 Application — une photo (éditeur)

- Clic sur `Palette` : les 10 réglages prennent les valeurs prédites en < 400 ms (sinon spinner dans le bouton). Une seule entrée dans l’historique d’édition → `Ctrl/Cmd+Z` annule tout d’un coup.
- Les autres réglages (courbes, HSL, masques, recadrage, détail) ne sont **jamais** touchés.
- Si la photo a déjà des réglages de base manuels, on applique quand même (geste explicite sur une photo) : l’annulation suffit. Pas de confirmation (friction inutile).

### 2.4 Application — lot (bibliothèque)

```
┌──────────────────────────────────────────────────────┐
│ Appliquer mon style à 64 photos                  [X] │
│ Seuls les réglages de base changent. Recadrage,      │
│ masques, retouches locales et autres réglages sont   │
│ conservés.                                           │
│ ☑ Ignorer les photos déjà retouchées                 │
│   12 photos sélectionnées ont déjà des réglages de   │
│   base manuels.                                      │
│                                  [Annuler] [Appliquer] │
└──────────────────────────────────────────────────────┘
```

- La case `skipEdited` n’apparaît que si au moins une photo est concernée ; **cochée par défaut** (P2).
- Progression : la modale passe en `ProgressBlock` (`batch.progress`, bouton `Arrêter`). Arrêter garde ce qui est fait et le dit (`batch.stopped`).
- Fin : modale fermée, toast succès `batch.done` (+ `batch.skipped`, `batch.failed` si > 0) avec action **Annuler** qui restaure les réglages antérieurs de toutes les photos touchées.
- Les vignettes se rafraîchissent au fil de l’eau (comportement existant de `ApplyAutoAdjustmentsToPaths`).

### 2.5 « Proposé par le style » vs réglages manuels

Dans `ControlsPanel`, pour chaque `Slider` des 10 réglages :

- Valeur issue du style et non retouchée → petit marqueur à droite du libellé : pastille 6 px `bg-accent` **+** micro-libellé `style.provenance.badge` (« Style ») en `TextVariants.small text-text-secondary`, tooltip `provenance.tooltip`. Lecteur d’écran : suffixe « proposé par mon style » dans le nom accessible du curseur.
- Dès que l’utilisateur bouge le curseur (ou double-clic reset), le marqueur disparaît : la valeur devient manuelle.
- En tête de la section Base : ligne `StatusNote info` « Mon style appliqué » ou `provenance.partial` (N réglages modifiés par vous) + lien `provenance.remove`. « Retirer mon style » restaure les valeurs d’avant l’application **pour les réglages encore marqués Style** uniquement (les réglages repris à la main ne bougent pas) ; toast `provenance.removed`.
- Données nécessaires (décision Sam) : dans les ajustements de l’image, `styleProvenance: { modelId, appliedAt, keys: string[], previous: Record<key, number> }`. Une clé sort de `keys` dès qu’elle est modifiée à la main. Persisté dans le sidecar pour survivre au redémarrage ; ignoré par copier/coller de réglages (le collé devient manuel).

### 2.6 Erreurs — style

| Cas | Présentation |
|---|---|
| Aucun style | popover / modale `noModel` + `train` (§2.1) |
| Modèle d’une version incompatible | `StatusNote warning` `errors.modelOutdated` + bouton `retrain` à la place de l’action |
| Fichier modèle illisible | `errors.modelDamaged` + `retrain` |
| Image illisible dans un lot | comptée dans `batch.failed` ; détail par nom (`errors.imageUnreadable`) |
| Échec d’inférence sur une photo (éditeur) | toast erreur `errors.generic` ; réglages inchangés |
| Corpus trop petit | §2.2 |
| Échec d’entraînement | `training.failed` + `failedBody` + `Réessayer` + `copyDetails` |

### 2.7 Critères de réception — style

1. Sans style entraîné, chaque point d’entrée mène à « Entraîner mon style… » ; aucune entrée ne reste grisée sans explication.
2. Corpus de 12 photos : lancement impossible, message `tooSmall`. Corpus de 30 : avertissement `small`, lancement possible.
3. Annuler l’entraînement laisse l’ancien style actif et utilisable.
4. Éditeur : après application, seuls les 10 réglages de base ont changé ; `Ctrl/Cmd+Z` une fois restaure l’état antérieur.
5. Les curseurs issus du style portent « Style » (texte, pas seulement la couleur) ; le marqueur disparaît après un déplacement manuel.
6. « Retirer mon style » ne touche pas un réglage repris à la main.
7. Lot de 64 photos dont 12 retouchées, case par défaut : 52 modifiées, 12 intactes ; « Annuler » du toast restaure les 52.
8. Le marqueur de provenance survit à un redémarrage de l’application.

---

## 3. Publication PicPortal

### 3.1 Existant et écarts

| Existant | Écart | Sévérité |
|---|---|---|
| Destination « Exporter vers PicPortal » dans `ExportPanel`, `PicPortalPanel` rendu **sous** le bouton d’export | Ordre de lecture inversé : on voit l’action avant les choix | Majeur |
| Champs e-mail / mot de passe avec placeholder seul | P5 | Majeur |
| Formulaire de création de galerie toujours déplié, 4 `select` vides obligatoires | Charge cognitive (Hick) sur chaque export | Majeur |
| `status` = `String(error)` brut (ex. `PicPortal login failed: …`) | P4 | Bloquant |
| Annulation = survol du bouton rouge (`group-hover`) | Indécouvrable, inaccessible au clavier/tactile | Bloquant |
| Échec partiel : texte dans le bouton, pas d’action de réessai dédiée | Réessai implicite en recliquant | Majeur |
| Succès : bouton vert, sans lien vers la galerie | Fin sans prolongement (pic-fin) | Mineur |
| Consentement facial : case non cochée par défaut, réinitialisée au changement de galerie | **Bon, on garde** ; texte à rendre explicite | Mineur |

### 3.2 Structure du panneau (destination PicPortal)

Dans `ExportPanel`, quand `destinationType === 'picportal'` : le bloc PicPortal remonte **dans la section Destination**, juste sous le sélecteur de destination. Trois étapes numérotées visuellement (région commune : chaque étape dans une carte `rounded-md border border-border-color p-3`, espacement `space-y-3`). Les sections Réglages de fichier, Dimensionnement, Métadonnées, Filigrane restent là où elles sont. Le bouton d’action reste en pied de panneau.

```
Destination  [ Dossier personnalisé | Dossier d’origine | ■Publier sur PicPortal ]

① Compte
   Connecté en tant que Camille Martin                      [Se déconnecter]
   camille@exemple.fr · Session conservée en sécurité…

② Galerie
   [ Mariage Dupont — 212 photos · Recherche par visage ▾ ] [⟳]
   [+ Nouvelle galerie]

   Données de recherche par visage
   ☐ Envoyer les données de recherche par visage pour ces photos
     Des signatures de visage sont calculées sur cet ordinateur…
     Les photos seront publiées sans données de visage.

③ Photos
   64 photos sélectionnées · Modifier la sélection
   ─────────────────────────────────────────────
   Taille estimée : 412 Mo
   [        Publier 64 photos        ]
```

### 3.3 Connexion

- Déconnecté : étape ① seule active, ② et ③ affichées repliées en `text-text-secondary` (on voit où l’on va). Champs avec **libellés visibles** `email`, `password` (+ bouton œil `showPassword/hidePassword`), `autoComplete` inchangés. `Entrée` dans le mot de passe = Se connecter.
- En cours : bouton `signingIn`, champs désactivés.
- Restauration au montage : ligne `restoring` avec squelette ; pas de clignotement du formulaire (n’afficher le formulaire que si la restauration échoue).
- Connecté : `connectedAs` + e-mail + `sessionSaved` ou `sessionNotSaved` (plateforme sans trousseau, `StatusNote warning`). Déconnexion = lien texte (plus une icône seule sans libellé).
- Erreurs, sous les champs (`StatusNote error`, `role="alert"`) : `badCredentials`, `offline`, `timeout`, `server`. Mot de passe vidé seulement sur `badCredentials` ; e-mail conservé.

### 3.4 Galerie

- `Dropdown` existant (ou `select` natif stylé comme aujourd’hui) ; chaque option : titre, `galleryPhotoCount`, et « Recherche par visage » si activée. Tri : dernières modifiées d’abord (si l’API le fournit, sinon ordre serveur). Au-delà de 12 galeries, champ de filtre en tête de liste.
- Aucune galerie : `noGalleries` + bouton `newGalleryButton` en primaire.
- **Nouvelle galerie** : bouton secondaire qui déplie le formulaire en place (divulgation progressive), focus sur le titre. Champs avec libellés, **valeurs par défaut** pour réduire les choix : Type = Événement, Accès = Toute personne ayant le lien, Visibilité = Brouillon, Recherche par visage = Désactivée (défaut protecteur de la vie privée, choix explicite pour l’activer). Champs conditionnels : client (nom, e-mail) si Type = Client ; mot de passe de galerie si Accès = Protégé. Validation en ligne au blur (`validation.*`), bouton `createGallery` → `creatingGallery`. Succès : formulaire replié, galerie sélectionnée, `galleryCreated` en `StatusNote success` 4 s. Échec : `errors.galleryCreateFailed`, champs conservés.
- Galerie supprimée côté serveur (404 à l’envoi) : `errors.galleryNotFound` + actualisation automatique de la liste, sélection vidée.

### 3.5 Consentement facial

- Visible seulement si la galerie choisie a la recherche par visage ; sinon une ligne `faceAnalysisUnavailable` en `text-text-secondary`.
- Case **non cochée par défaut**, réinitialisée à chaque changement de compte ou de galerie (comportement actuel de `picPortalDestinationChanged` conservé).
- Texte `faceConsent.body` toujours visible (pas dans un tooltip). Sous la case, conséquence en direct : non cochée → `faceConsent.off`.
- Non bloquant : publier sans données de visage est un chemin normal, pas un état d’erreur. Aucune relance, aucune formulation culpabilisante.

### 3.6 Sélection des images

- Source = sélection courante de la bibliothèque (comportement actuel `numImages`). Ligne `selectedImages` + lien `changeSelection` (ramène à la grille, panneau d’export conservé).
- 0 photo : `noSelection`, bouton désactivé.
- Formats : `unsupportedFormat` / `unsupportedMasks` en `StatusNote warning` **au-dessus** du bouton, avec la cause précise.
- Bouton désactivé → son libellé dit ce qui manque (`missingStep.account`, `missingStep.gallery`) au lieu d’un bouton grisé muet.

### 3.7 Progression et annulation

```
Préparation des photos · 12 sur 64
██████░░░░░░░░░░░░░░░░░░░░░░░░░░
Gardez PicPortal Editor ouvert jusqu’à la fin de l’envoi.
                                               [Annuler]
```

- `ProgressBlock` à la place du bouton principal. Deux étapes (`progress.rendering` puis `progress.uploading`) d’après `stage` de `picportal-progress`. Le formulaire au-dessus passe en lecture seule (`disabled`, déjà en place).
- **Annuler** : bouton secondaire toujours visible. Clic → `cancelling` ; fin → `StatusNote info` `cancelled` avec le nombre déjà envoyé. Pas de confirmation (le travail fait reste acquis et la reprise est idempotente).
- Session expirée **pendant** l’envoi (`picportal-session-invalidated`) : l’envoi s’arrête, étape ① se rouvre avec `reconnectToResume` ; après reconnexion, bouton `resume` qui reprend la file (clé d’idempotence existante).

### 3.8 État final

- **Succès total** : `StatusNote success` `success` (nombre + titre de galerie) + `openGallery` (ouvre l’URL publique dans le navigateur système ; URL construite à partir du `slug` — Sam confirme le format) + `newPublication` (remet le bouton à l’état initial). L’état succès persiste jusqu’à changement de sélection ou de galerie.
- **Échec partiel** : `StatusNote warning` `partialFailure` + primaire `retryFailed` (renvoie uniquement les échecs) + `showDetails` (liste nom de fichier → raison traduite).
- **Échec total** : `StatusNote error` avec le message mappé + `Réessayer` + `copyDetails`.

### 3.9 Erreurs réseau et authentification (mapping code → texte)

| Cause backend (exemples actuels) | Code proposé | Texte | Action proposée |
|---|---|---|---|
| 401 à la connexion | `bad_credentials` | `errors.badCredentials` | corriger |
| 401/419 en cours de session, refresh raté | `session_expired` | `sessionExpired` / `reconnectToResume` | se reconnecter, Reprendre |
| échec d’effacement du trousseau | `session_clear_failed` | `sessionInvalidationFailed` | Se déconnecter |
| DNS / connexion refusée / hors ligne | `offline` | `errors.offline` | Réessayer |
| délai dépassé | `timeout` | `errors.timeout` | Réessayer |
| 5xx | `server` | `errors.server` | Réessayer |
| 403 | `forbidden` | `errors.forbidden` | choisir une autre galerie |
| 404 galerie | `gallery_not_found` | `errors.galleryNotFound` | actualisation auto |
| 413 | `file_too_large` | `errors.fileTooLarge` | réduire la taille |
| « local derivative processing is disabled by the server » | `derivatives_disabled` | `errors.derivativesDisabled` | — (remonter à Maxime) |
| « another PicPortal publication is already running » | `already_running` | `errors.alreadyRunning` | — |
| rendu local raté | `render_failed` | `errors.renderFailed` | Réessayer |
| autre | `unknown` | `errors.generic` | Réessayer + copier le détail |

Le backend renvoie aujourd’hui des chaînes ; il faut une erreur structurée `{ code, message, path? }` (décision d’API : Sam). Transitoire accepté : mapping côté front par motif de chaîne, isolé dans `utils/picPortalErrors.ts`.

### 3.10 Critères de réception — publication

1. Le bloc PicPortal apparaît au-dessus du bouton d’action, en trois étapes lisibles dans l’ordre.
2. Aucun champ sans libellé visible ; aucun message brut du backend à l’écran.
3. Mauvais mot de passe → `badCredentials` sous les champs, e-mail conservé.
4. Création de galerie : une galerie « Événement / lien / brouillon / visages désactivés » se crée en saisissant **uniquement le titre**.
5. Consentement facial décoché par défaut, réinitialisé au changement de galerie, texte d’explication visible sans survol.
6. « Annuler » visible et atteignable au clavier pendant tout l’envoi ; après annulation, le nombre de photos déjà envoyées est affiché.
7. Réseau coupé au milieu : `partialFailure` + « Réessayer N photos » ; après retour réseau, seules les N photos sont renvoyées, sans doublon dans la galerie.
8. Session expirée pendant l’envoi : reconnexion puis « Reprendre » finit l’envoi sans doublon.
9. Succès : lien « Ouvrir la galerie » ouvre la bonne galerie.

---

## 4. Textes FR et EN

- Fichiers : `docs/ux/i18n/en.json`, `docs/ux/i18n/fr.json` — 271 entrées, même arborescence que `src/i18n/locales/*.json`, fusion profonde vérifiée sans conflit de type (en : 178 nouvelles clés, 93 mises à jour ; fr : 184 / 100).
- Les clés existantes réutilisées gardent leur nom ; leurs valeurs sont réécrites (ton, clarté, FR manquant). Les ~60 clés `modals.culling.*` encore en anglais dans `fr.json` sont traduites.
- Pluriels : `_one` / `_other` ; en français `_many` = `_other` (convention déjà présente dans `fr.json`). Après fusion, **supprimer les variantes `_few` obsolètes** de `fr.json` pour les clés réécrites (`analyzingCount`, `photosInFolder`, `failedAnalysisCount`, `persistenceFailure`, `preservedCount`).
- Typographie FR : apostrophe typographique (’), guillemets « » avec espace insécable à ajouter par l’intégration si l’outil le permet, points de suspension « … ».
- Les 11 autres langues : fusion des clés EN comme repli (comportement i18next actuel) ; traduction hors périmètre v1.
- Vérification attendue du développeur : `npm run i18n:runtime-check` et `npm run typecheck` au vert après fusion.

## 5. Hors périmètre v1 (proposé, non spécifié)

- Badge « Style » sur les vignettes de la bibliothèque.
- Filtre « Uniquement les photos retenues » dans l’étape Photos de la publication (raccourci tri → publication).
- Reprise des propositions de tri après redémarrage (aujourd’hui en mémoire dans `useUIStore`).

## 6. Limites de ce document

- Maquettes en schémas texte ; aucune capture : l’application native n’a pas été rendue pendant cette exécution. La porte de vérité visuelle s’appliquera à la revue d’implémentation (captures 1440×900 exigées à Théo/Sam, vérification Camille).
- Seuils du corpus (20 / 50), niveaux de fiabilité, format d’URL de galerie et codes d’erreur structurés sont des propositions à confirmer par Sam.
