# Débits routiers Avatar (Cerema) autour des autoroutes

Récupérés le 17/10/2026 sur l'API publique d'Avatar (`https://avatar.cerema.fr/api/`, sans compte).

## Fichiers

| Fichier | Contenu |
|---|---|
| `compteurs.csv` | 1 301 compteurs (un par point de comptage et par sens) au format de `irve/README.md` : `compteur_id, lat, lon, axe_rattache, sens, pk_projete, delai_rattachement_min`, plus des colonnes descriptives |
| `comptages.csv` | 7,08 M lignes horaires du 1/03/2026 00:00 au 17/10/2026 18:00 (heure de Paris, sans fuseau, comme `occupation_bornes.csv`) : `horodatage, compteur_id, debit_vh, vitesse_kmh, part_estimee_pct` |
| `brut/` | liste complète des 9 754 points Avatar (json.gz) et les points retenus |
| `scripts/` | `cand.py` (rattachement aux axes), `dl.py` puis `dl3.py` (téléchargement par lots de 40 ; dl3 pour mars-avril et août-octobre), `build.py` (mise au format) |

## Choix faits

- **Sélection** : point à 5 km au plus de la polyligne d'un axe de `axes_approx.csv`, route d'importance BD TOPO 1 ou 2, point non supprimé dans Avatar. 1 979 points retenus, 1 301 ont des mesures ; 1 297 couvrent le 9 au 15 juillet. Les points marqués « supprimés » dans Avatar n'ont aucune mesure (vérifié sur la N165 et la N24).
- **pk_projete** : projection sur la polyligne de `axes_approx.csv` (même référentiel que `stations.csv`).
- **sens** : déduit de l'orientation du tronçon mesuré (`linear_section`) projeté sur l'axe : 1 = pk croissants, 2 = décroissants, 0 = route perpendiculaire ou trop courte pour conclure (867 compteurs). Vérifié sur les libellés « Vers … » des exploitants (ex. A7 « Vers Marseille » → 1, A31 « Vers Luxembourg » → 2).
- **delai_rattachement_min** : 0 si la route porte le numéro de l'axe, sinon distance à l'axe / 70 km/h.
- **part_estimee_pct** : part de l'heure reconstituée par le modèle d'Avatar au lieu d'être mesurée (100 = entièrement estimée). À filtrer ou à pondérer à l'apprentissage ; `part_mesuree_9_15_juillet_pct` dans `compteurs.csv` la résume sur la semaine d'occupation.

## Limite importante

Avatar ne couvre que le réseau non concédé (DIR, métropoles). Les compteurs se concentrent donc **aux extrémités urbaines des autoroutes** (Paris, Lyon, Bordeaux) ; la colonne `hors_agglo_extremites` repère les 148 compteurs à plus de 30 km des deux bouts d'axe. Sur les axes demandés :

- **A10** : 107 compteurs, dont 8 interurbains (N254 Allaines pk 84, N10 pk 327, N248 pk 399, N141 pk 456).
- **A6** : 219 compteurs, quasi tous en Île-de-France ou à Lyon ; 4 interurbains (N6 et N65 vers Auxerre, pk 148-153).
- **A7** : 102 compteurs, dont 16 sur la N7 parallèle entre Vienne et Valence (pk 38 à 110) : la meilleure couverture. Rien au sud de Valence.
- **A9** : 4 compteurs seulement (N109 Juvignac, A75 Béziers), estimés à plus de 80 % : inutilisable en l'état.

Axes ajoutés le 17/10 (Bretagne et Alpes), téléchargés sur toute la période :

- **Bretagne** : A84 (39 compteurs dont 16 interurbains, la meilleure couverture), N137 (14, mesurés à 100 %), N12 (10), N165 (10, mais estimés à 96 % : peu utilisable). N24 n'a que des points supprimés, donc aucune donnée.
- **Alpes** : A43 (27, tous à Lyon au départ de l'axe), A41 (9, autour de Grenoble), A48 (8, Grenoble), A49 (2), N90 (8, Albertville-Moûtiers). A51 n'a que des points supprimés ; rien sur A40 ni N85.

Pour les sections concédées (A9, A7 sud, A10 et A6 en rase campagne), il faudrait les comptages des sociétés d'autoroute (ASFA, Vinci, APRR), ou se reposer sur la prévision Bison Futé.

## Mise à jour

`python3 scripts/dl.py brut/points_selectionnes.json <dossier_brut>` puis `python3 scripts/build.py` (chemins à adapter en tête de script). L'API limite le nombre de requêtes par minute (erreurs 429 gérées par réessai).
