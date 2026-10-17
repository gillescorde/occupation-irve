# Bison Futé des bornes de recharge : données, graphe et apprentissage

Ce dossier contient la première brique du projet : le schéma des données d'entrée, une manière
d'injecter le réseau routier et la propagation du trafic dans un modèle, et un script
d'apprentissage qui compare deux familles de modèles. La restitution (Flask) viendra ensuite et
lira les sorties de `entrainer.py`.

Tout tourne aujourd'hui sur des **données synthétiques** au même format que celui attendu. Il
suffira de remplacer le dossier de données par le vrai.

```
generer_donnees_synthetiques.py   fabrique un jeu de test au bon format
donnees.py                        chargement, conversion de l'export Qualicharge, agrégation par PDC
preparer_stations.py              place les stations sur les axes (axe, sens, pk) à partir d'un vrai tracé
reperer_stations_autoroute.py     repère les stations d'autoroute dans ton référentiel (tracé approché)
donnees_reelles/                  stations.csv (170 stations d'autoroute) et axes_approx.csv
graphe.py                         graphe routier, temps de parcours, variables de propagation
gnn.py                            réseau de neurones de graphe spatio-temporel (PyTorch)
entrainer.py                      chaîne complète : étape 1 trafic, étape 2 occupation, évaluation
resultats_synthetiques/           métriques et graphe obtenus sur le jeu synthétique
```

## 1. Les données d'entrée

Pas de temps conseillé : l'heure, ou 30 min si le volume le permet. Tous les fichiers sont des CSV.

| Fichier | Colonnes | Rôle | Statut |
|---|---|---|---|
| `occupation_bornes.csv` | horodatage, station_id, nb_pdc, nb_pdc_occupes, nb_pdc_hors_service | **cible** | tu l'as (6 mois) |
| `stations.csv` | station_id, lat, lon, axe, sens, pk, nb_pdc, puissance_kw | position sur le graphe | à compléter (axe/sens/pk) |
| `comptages.csv` | horodatage, compteur_id, debit_vh | trafic observé | à récupérer |
| `compteurs.csv` | compteur_id, lat, lon, axe_rattache, sens, pk_projete, delai_rattachement_min | position sur le graphe | à construire |
| `calendrier.csv` | date, ferie, vacances, debut_vacances, fin_vacances, bison_fute_couleur | variables exogènes | data.gouv + Bison Futé |
| `aretes.csv` (facultatif) | source, cible, temps_min | graphe explicite | sinon construit automatiquement |

### Ton historique : l'export de la collecte Qualicharge

L'extrait reçu le 17/10 (2 998 lignes, une par station, toutes collectées le 9 juillet 2026 à
11 h 56) vient de l'API `map.qualicharge.beta.gouv.fr/api/irve/points/`. Une ligne = l'état d'une
station : `station_key` (identifiant d'itinérance AFIREV), `fetched_at` (première collecte de cet
état), `last_seen_at` (dernière fois qu'il a été revu), `latest_status_at` (horodatage de l'opérateur)
et les compteurs `availability.*_count`. La fonction `donnees.convertir_snapshots_qualicharge()` le
transforme au format ci-dessous :

```python
import pandas as pd
from donnees import convertir_snapshots_qualicharge
brut = pd.read_csv("export_qualicharge.csv")
convertir_snapshots_qualicharge(brut, pas="1h").to_csv("donnees/occupation_bornes.csv", index=False)
```

Ce que l'extrait montre, et les choix faits en conséquence :

- libre + occupé + réservé + inconnu = `pdcs_with_dynamic_count` sur 100 % des lignes. Le taux
  d'occupation retenu est (occupé + réservé) / (libre + occupé + réservé) : les PDC à l'état
  **inconnu** (7 % des PDC de l'extrait) sont sortis du dénominateur au lieu d'être comptés libres.
- Hors service = `pdcs_with_dynamic_count − en_service_count` (16 % des stations en ont au moins un).
- **Fraîcheur** : `latest_status_at` a moins d'une heure pour 58 % des stations, mais plus de 24 h pour
  11 % et plus d'une semaine pour 5 %. Au-delà de 24 h (paramètre `statut_perime`), le statut est
  considéré comme une remontée morte et devient inconnu.
- Chaque état est supposé valable de `fetched_at` jusqu'à l'état suivant de la même station, au plus
  jusqu'à `last_seen_at` + 5 min. La conversion fonctionne que l'historique ait une ligne par
  collecte ou une ligne par changement d'état.
- Occupation moyenne dans l'extrait : 18 %, et 44 % des stations ont au moins un PDC occupé.
- **Pas de coordonnées** dans cet export : `preparer_stations.py` les reprend du fichier IRVE
  statique consolidé (jointure sur l'identifiant de station), puis projette chaque station sur le
  tracé des autoroutes pour obtenir axe, sens et pk.

### L'historique complet (dossier `qualicharge` sur ton PC)

`charging_station_availability_snapshots.csv` (350 Mo, 1,28 million de lignes, 7 388 stations) est un
journal des changements d'état, collecté toutes les 5 min. Il ne couvre que **du 9 au 15 juillet 2026**
(6 jours), avec deux trous de collecte (9 h le 9/07, 3 h 30 le 10/07). Converti pour les stations
d'autoroute : `donnees_reelles/occupation_bornes.csv` (150 stations × heure, couverture 99 %).

### Débits Avatar et rattachement au graphe

`rattacher_compteurs.py` relie les compteurs de `donnees_reelles/debits/` (Avatar, mai à juillet 2026)
aux stations d'autoroute, puis mesure, sur la semaine du 9 au 15 juillet, la corrélation entre le
trafic amont propagé et l'occupation. Les heures estimées à plus de 50 % par Avatar sont écartées.
La corrélation « brute » est gonflée par le cycle jour/nuit commun ; la corrélation « hors cycle
journalier » (chaque série moins sa moyenne par heure) mesure l'information en plus de l'heure.

| Compteurs utilisés | Stations reliées | r brute (médiane) | r hors cycle (médiane) |
|---|---|---|---|
| Tous (997) | 113 / 149 | 0,55 | 0,07 |
| Interurbains seulement (96, option `--interurbains-seulement`) | 71 / 149 | 0,58 | 0,17 |

Les compteurs urbains (Paris, Lyon, Bordeaux) diluent le signal : à garder seulement les
interurbains. Sorties dans `donnees_reelles/rattachement/` et `donnees_reelles/rattachement_interurbains/`.
Sur 6 jours, ces corrélations restent fragiles (environ 140 heures par station).

### Ton référentiel de stations : `charging_stations.csv`

Reçu le 17/10 : 7 388 stations de recharge rapide (50 kW et plus), avec coordonnées, nombre de PDC,
puissance et opérateur. Les 2 998 stations de l'extrait d'occupation y figurent toutes.
`reperer_stations_autoroute.py` en tire `donnees_reelles/stations.csv` :

- **170 stations d'autoroute** sur 20 axes (A10 : 21, A9 : 18, A7 : 18, A6 : 15...). Les bornes
  réservées aux poids lourds sont écartées.
- **Axe** : lu dans le nom pour 119 stations (« Electra A10 - Aire de Cézac »), déduit de la proximité
  d'un axe pour 51 noms d'aire sans numéro.
- **pk** : projection sur un tracé approché par villes étapes (codé dans le script, coordonnées à
  quelques km près). Erreur de quelques km, sans conséquence pour un graphe d'aires espacées de 20 à 60 km.
- **Sens** : lu dans le nom (« direction Paris », « Sens Bordeaux Paris ») ou déduit du suffixe
  Nord/Sud/Est/Ouest et du côté de la chaussée. La règle du côté de la chaussée donne le même sens
  que la direction écrite sur toutes les stations qui portent les deux. Reste 0 (inconnu) pour 85
  stations : le graphe les rattache aux deux sens.
- **Limite** : la détection passe par le nom. Une station d'aire nommée seulement par sa commune ou
  son opérateur (« Tesla Supercharger Beaune ») n'est pas trouvée. Un vrai tracé d'autoroute avec
  `preparer_stations.py` les rattraperait à la distance.
- Dans l'extrait du 9 juillet, 72 de ces stations ont un statut. Leur occupation moyenne est de 25 %,
  contre 19 % pour les autres stations, et elles sont plus grosses (12 PDC contre 5).

Remarques pratiques :

- **Taux d'occupation** = occupés / (total − hors service). Les PDC en panne ne doivent pas faire
  baisser artificiellement le taux.
- Si ton historique est au **niveau du point de charge** (format IRVE dynamique d'Etalab :
  `id_pdc_itinerance`, `etat_pdc`, `occupation_pdc`, `horodatage_derniere_maj`), la fonction
  `donnees.agreger_statuts_pdc()` le ramène au format station × heure. Elle considère que chaque
  statut reste valable jusqu'au suivant et calcule la fraction de temps occupée.
- **Axe, sens et pk** sont la clé de tout le reste : ils placent stations et compteurs sur le
  graphe. Convention retenue : sens 1 = pk croissants. Une aire desservant les deux sens se
  dédouble en deux stations.
- **Compteurs hors autoroute** : un compteur sur une nationale parallèle ou une pénétrante se
  « rattache » à l'axe à son pk projeté, avec un délai de rattachement en minutes. Il ne mesure
  pas le trafic autoroutier, il en donne un **indicateur** corrélé. C'est suffisant parce que les
  débits sont normalisés par compteur (1 = trafic moyen de ce compteur) : le modèle utilise les
  variations, pas les niveaux.
- **Sources utiles** : le calendrier scolaire et les jours fériés sont sur data.gouv.fr. La
  prévision Bison Futé officielle (couleur par jour et par sens) est probablement la meilleure
  variable exogène à J+1/J+7. Le TMJA du réseau routier national donne un niveau de trafic moyen
  sur les tronçons sans compteur.

## 2. Comment le réseau routier entre dans le modèle

### Le graphe

Chaque **station** et chaque **compteur** est un nœud. Les arêtes sont orientées dans le sens de
circulation :

- le long d'un axe et d'un sens, chaque station est reliée à la suivante ;
- chaque compteur est relié à la première station en aval de son point de rattachement.

Le poids d'une arête est le **temps de parcours** (distance / 110 km/h + délai de rattachement).
Un plus court chemin (Dijkstra) donne ensuite, pour tout couple de nœuds, le temps τ pour aller de
l'un à l'autre. Il vaut l'infini si le second n'est pas en aval du premier.

Pour un réseau qui se croise (A10/A71, A6/A77...), il faudra remplacer la construction automatique
par un `aretes.csv` tiré d'OpenStreetMap ou de Route 500. Le reste du code ne change pas.

### Propagation du trafic : deux manières de l'exploiter

**a) À la main, pour un modèle classique (gradient boosting).** Pour chaque station s et chaque
heure t, on calcule :

```
trafic_amont_propage(s, t) = Σ_c w_cs · debit_c(t − τ_cs)  /  Σ_c w_cs      avec  w_cs = exp(−τ_cs / 90 min)
```

On prend les compteurs c situés en amont à moins de 5 h, on décale leur débit du temps qu'il faut
pour atteindre la station, et on pondère par la proximité. On calcule aussi la même variable
décalée de 1, 2 et 3 h supplémentaires (inertie de la file d'attente) et le compteur amont le plus
proche seul. À cela s'ajoute l'occupation des stations voisines amont/aval connue il y a H heures.

**b) Appris, dans un réseau de neurones de graphe (GNN).** C'est l'équivalent d'un perceptron dont
les poids sont partagés par tous les nœuds, et qui à chaque couche mélange l'état d'un nœud avec
celui de ses voisins :

```
H' = σ( H·W₀ + A·H·W₁ + A²·H·W₂ )         H : [N nœuds × d variables]
```

A est construite à partir des temps de parcours (A[j,i] = exp(−(τ_ij/60)²), normalisée par ligne).
Le produit A·H est donc la moyenne pondérée des voisins amont. Le modèle (`gnn.py`, dans l'esprit
de STGCN et Graph WaveNet) enchaîne deux fois :

1. une **convolution temporelle causale** sur une fenêtre de 6 h par nœud (un TDNN), qui apprend
   les retards de propagation ;
2. une **convolution de graphe** avec trois matrices : A_amont (le trafic arrive), A_aval (une aire
   saturée reporte la demande) et une matrice **apprise** `softmax(relu(E₁E₂ᵀ))` qui corrige ce que
   la carte ignore (compteur mal rattaché, report vers une aire hors autoroute...).

Un plongement appris par nœud capte enfin les spécificités de chaque station (attractivité, taille).

## 3. La chaîne d'apprentissage (`entrainer.py`)

```
calendrier + historique compteurs ──► ÉTAPE 1 : prévision du débit aux compteurs à H (gradient boosting)
                                                   │
graphe (temps de parcours) ───────────────────────┤
occupation connue il y a ≥ H ─────────────────────┤
calendrier, station ──────────────────────────────▼
                                    ÉTAPE 2 : occupation de chaque station à t
                                    (gradient boosting avec variables de graphe  OU  GNN)
                                                   │
                                                   ▼
                          taux prévu + couleur  vert < 40 % ≤ orange < 70 % ≤ rouge < 90 % ≤ noir
```

Points de méthode :

- **Découpage temporel** 70 / 15 / 15 %, jamais aléatoire : un découpage aléatoire laisserait le
  modèle voir le lendemain de chaque heure de test et ferait croire à de faux bons scores.
- **Horizon H** (`--horizon-h`, 24 h par défaut) : seules les occupations passées d'au moins H sont
  utilisées, pour que l'évaluation corresponde à une vraie prévision.
- **Double évaluation** de l'étape 2 : avec le trafic *observé* (borne haute, comme si l'étape 1
  était parfaite) et avec le trafic *prévu* par l'étape 1 (ce qu'on aura en production).
- **Deux références** pour juger honnêtement : le profil moyen station × jour × heure, et un
  gradient boosting sans aucune variable de trafic.
- L'étape 2 est apprise sur le trafic observé. Si l'écart observé/prévu pèse trop en production, on
  peut l'apprendre sur des prévisions hors échantillon de l'étape 1.

## 4. Lancer

```bash
pip install -r requirements.txt
python generer_donnees_synthetiques.py --sortie donnees_synthetiques
python entrainer.py --donnees donnees_synthetiques --horizon-h 24 --sortie resultats
#   options : --pas 30min, --horizon-h 168, --sans-gnn, --epochs 20
```

Environ 3 minutes sur CPU pour 32 stations, 12 compteurs et 6 mois horaires. Les sorties dans
`resultats/` sont `metriques.csv`, `predictions_test.csv` (station, heure, modèle, prévu, réel,
couleur : c'est ce que lira l'interface), `modeles_hgb.joblib`, `modele_gnn.pt`, `config.json` et
`aretes_utilisees.csv` (le graphe effectivement utilisé, à vérifier sur une carte).

## 5. Résultats sur le jeu synthétique (J+1, test = août)

| Modèle | Erreur moyenne (points) | Couleur exacte | Alertes rouge/noir détectées | Alertes justes |
|---|---|---|---|---|
| Profil moyen station × jour × heure | 10,4 | 73 % | 18 % | 96 % |
| Gradient boosting sans trafic | 7,3 | 82 % | 64 % | 75 % |
| Gradient boosting, trafic observé | 6,2 | 85 % | 67 % | 84 % |
| Gradient boosting, trafic prévu | 7,3 | 81 % | 73 % | 68 % |
| GNN, trafic observé | 6,1 | 85 % | 72 % | 79 % |
| GNN, trafic prévu | 7,4 | 81 % | 80 % | 63 % |

Ce que ça montre, avec la prudence qu'impose un jeu synthétique :

1. **Le trafic aide vraiment quand il est connu** (6,2 contre 7,3 points). C'est le cas d'usage
   « estimer maintenant » ou « dans 1 à 3 h » à partir des compteurs temps réel.
2. **À J+1, le gain disparaît si la prévision de trafic ne contient que le calendrier** : l'étape 2
   le sait déjà. La chaîne en deux étapes ne vaut que si l'étape 1 apporte une information en plus :
   prévision Bison Futé officielle, événements, météo. C'est le point le plus important pour la suite.
3. **GNN et gradient boosting font jeu égal** sur ce petit réseau. Le GNN détecte un peu mieux les
   saturations, au prix de plus de fausses alertes. Avec 6 mois et quelques dizaines de stations,
   je commencerais par le gradient boosting (plus simple, explicable, robuste) et je garderais le
   GNN pour quand le réseau couvert grandira.
4. **6 mois, c'est court** : le test d'août n'a vu qu'un été partiel en apprentissage. Avec une
   seule saison estivale, la validation croisée par blocs de semaines sera plus fiable qu'un seul
   découpage.

## 6. Ce qu'il faut pour passer aux vraies données

1. ~~Un extrait du CSV d'occupation~~ : reçu, conversion en place (voir section 1).
2. ~~Le référentiel des stations~~ : reçu, 170 stations d'autoroute placées (tracé approché). Un vrai
   tracé des autoroutes (axe, pk, lat, lon) améliorerait la détection et les pk.
3. La liste des compteurs Avatar disponibles, avec un historique couvrant les mêmes 6 mois.
4. Si possible, l'historique des prévisions Bison Futé par jour et par sens.
