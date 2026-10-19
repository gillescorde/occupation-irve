# To-do de lundi : Qualicharge et HERE

Tout ce qui ne dépend ni de Qualicharge ni de HERE est prêt dans le dépôt
[occupation-irve](https://github.com/gillescorde/occupation-irve) (détail dans `note_technique.md`) :

- **Calendrier 2016-2026** : `donnees_reelles/calendrier/calendrier.csv`. Une ligne par jour : vacances par zone,
  fériés, ponts et Bison Futé par sens et par zone. 2020 manque.
- **Stations** : `donnees_reelles/stations.csv`, 514 stations sur 33 axes (193 sur aire, 321 près d'une
  sortie). Carte dans `cartes/axes_stations.png`.
- **Compteurs Avatar** : `donnees_reelles/debits/`, 1 301 compteurs du 1er mars au 17 octobre 2026,
  rattachés au graphe dans `rattachement*/`.

## 1. Qualicharge (le matin)

1. ~~**Récupérer l'export complet**~~ : reçu, trois exports mensuels de **juillet à septembre** (pas de mars),
   rangés dans `data_IRVE/Qualicharge` (1,2 à 1,6 Go chacun). L'apprentissage portera donc sur l'été :
   l'hiver et le printemps manqueront, et la validation par blocs de semaines devient indispensable.
2. **Référentiel** : `charging_stations.csv` (9,8 Mo) est aussi dans `data_IRVE/Qualicharge`. Relancer la
   sélection si des stations ont ouvert depuis juillet :
   ```bash
   python reperer_stations_autoroute.py --referentiel "$DATA/Qualicharge/charging_stations.csv" `
       --sortie donnees_reelles/stations.csv --sortie-axes donnees_reelles/axes_approx.csv
   ```
   avec, en PowerShell, `$DATA = "C:/Users/corde/OneDrive - ifpen.fr/Documents/TECHNIQUES/data_IRVE"`
   (dépendances : `pip install -r requirements.txt` dans l'environnement Anaconda).
3. **Compresser puis convertir** avec `preparer_qualicharge.py`, qui lit les exports par morceaux, ne garde
   que les stations retenues, met les mois bout à bout et passe au pas horaire :
   ```bash
   python preparer_qualicharge.py --compresser --snapshots "$DATA/Qualicharge/charging_station_availability_snapshots_*.csv"
   python preparer_qualicharge.py --snapshots "$DATA/Qualicharge/charging_station_availability_snapshots_*.csv.xz" `
       --stations donnees_reelles/stations.csv --sortie donnees_reelles/occupation_bornes.csv.xz
   ```
   La compression divise la taille par 10 environ (essai sur l'échantillon). Une fois les .xz vérifiés,
   les .csv peuvent être supprimés (`--supprimer-csv`). Tous les scripts lisent les .xz tels quels.
4. **Contrôler** : la conversion affiche les trous de collecte, la part de PDC au statut connu et les
   stations dont le nombre de PDC a changé (une extension fait baisser le taux sans que le trafic
   bouge). Les noter dans `.notes/inputs.md`.
5. **Refaire le rattachement** des compteurs, cette fois sur toute la période :
   ```bash
   python rattacher_compteurs.py --donnees donnees_reelles --avatar "$DATA/avatar" --interurbains-seulement `
       --sortie donnees_reelles/rattachement_interurbains
   ```
   Il faut regarder la corrélation **hors cycle**. Si elle reste autour de 0,1 à 0,2, Avatar n'apportera
   presque rien et HERE devient la seule source de trafic.

## 2. HERE (l'après-midi)

1. **Clarifier le contrat**. Il faut savoir :
   - quel historique est disponible : vitesses au pas de 15 min avec vitesse fluide et nombre de sondes
     ou confiance, et sur quelle profondeur ;
   - si Traffic Patterns intègre les jours particuliers ;
   - quels sont les quotas de l'API Traffic Flow.
2. **Extraire l'historique**. Il couvre les 33 axes dans les deux sens, en classes fonctionnelles 1 et 2
   (les nationales bretonnes et alpines sont en classe 2). Il va de mars 2026 à aujourd'hui, et
   idéalement sur 2024-2025 pour couvrir les vacances d'hiver. Il faut aussi le **référentiel des
   segments** : identifiant, géométrie, sens.
3. **Rattacher les segments aux axes** avec les fonctions de `here_flux.py` : `placer_sur_axe`, puis
   `agreger_troncons` par tranche de 10 km. Le résultat est une table horodatage × axe × sens × tronçon
   avec la vitesse, la congestion, le temps de parcours et le nombre de sondes.
4. **Démarrer la collecte temps réel**. Mettre la clé dans la variable d'environnement `HERE_API_KEY`,
   jamais dans le code. Lancer ensuite `here_flux.py` toutes les 15 min en tâche planifiée, pour
   constituer l'historique de production.
5. **Écrire `donnees_reelles/here/here_troncons.csv`** à partir de l'historique, au format de
   `here_flux.py` : horodatage, axe, sens, troncon_pk, vitesse_kmh, vitesse_fluide_kmh, congestion,
   jam_factor, et `indice_volume` (ou `nb_sondes`) si HERE le fournit. `entrainer.py` est déjà prêt
   (`--trafic here | here+avatar | avatar`) et testé sur un HERE simulé : il ne restera qu'un script de
   conversion, à écrire quand on verra le format exact de l'extraction.

## 3. Premier apprentissage réel (à partir de mardi)

1. Faire l'étape 1, la prévision du trafic HERE par tronçon, à J+1, à partir du calendrier, de Bison
   Futé et de Traffic Patterns.
2. Faire l'étape 2 en gradient boosting d'abord, puis le GNN, sur le taux d'occupation, avec une
   **validation par blocs de semaines** : un seul été dans les données, donc pas de découpage unique.
   Lancer deux fois, `--trafic here` puis `--trafic here+avatar` : si Avatar n'améliore pas l'erreur,
   il ne sert plus qu'à caler l'indice de volume HERE.
3. Construire les **journées types**, avec une médiane et un P90 par station et par type de jour : c'est la
   référence à battre et l'outil de planification.
4. Rejouer les scénarios de pénétration (`scenarios.py`) sur l'historique long.

## 4. En suspens (pas bloquant)

- **Tracés OpenStreetMap** : l'accès est bloqué depuis le cloud. On peut le lancer depuis le
  poste du bureau :
  ```bash
  python telecharger_osm.py --sortie donnees_reelles/osm/brut
  python traces_osm.py --brut donnees_reelles/osm/brut --sortie donnees_reelles
  python reperer_stations_autoroute.py --referentiel charging_stations.csv --traces donnees_reelles/axes_osm.csv \
      --echangeurs donnees_reelles/echangeurs.csv --sortie donnees_reelles/stations.csv
  ```
- **Bison Futé 2020** : aucun calendrier n'a été trouvé. On peut s'en passer, puisque c'est l'année
  du Covid.
- **Relecture des stations de sortie** : vérifier sur la carte celles qui servent surtout le trafic
  local, avant de les garder dans l'apprentissage.
