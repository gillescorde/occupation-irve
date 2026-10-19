# Données HERE pour le « Bison Futé des bornes »

*Note technique, 17 octobre 2026. Ce qu'il faut récupérer chez HERE, comment le modèle l'utilise,
et comment se brancher sur l'API Traffic Flow. Code associé : `here_flux.py`.*

> Les noms de produits, de champs et les limites d'API ci-dessous viennent de la documentation
> publique de HERE telle que je la connais. Je n'ai pas pu les vérifier sur votre contrat :
> l'environnement de calcul n'a pas accès à HERE. Les points marqués **(à vérifier)** sont à
> confirmer avec votre interlocuteur HERE ou dans votre portail.

## 1. Pourquoi HERE

Les compteurs Avatar ne couvrent que le réseau non concédé. Ils sont surtout aux extrémités urbaines
des autoroutes et ne voient presque rien des sections à péage. Sur la semaine du 9 au 15 juillet,
une fois le cycle jour/nuit retiré, leur lien avec l'occupation des aires reste faible : corrélation
médiane de 0,17 avec les seuls compteurs interurbains.

HERE mesure, par véhicules sondes, la vitesse **sur l'autoroute elle-même**, segment par segment,
y compris en rase campagne. Il n'y a pas de débit, mais trois informations utiles :

| Information | Ce qu'elle apporte au modèle |
|---|---|
| Vitesse et vitesse fluide | un indice de congestion, et le **temps de parcours réel** entre deux points du graphe |
| Jam factor (0 à 10) | le même indice, calculé par HERE |
| Nombre de sondes (historique seulement) | un indicateur **relatif** du volume, qui voit un samedi de départ chargé mais encore fluide |

## 2. Ce qu'il faut récupérer

### 2.1 Pour l'apprentissage : l'historique

| Priorité | Produit HERE | Contenu attendu | Usage |
|---|---|---|---|
| 1 | **Traffic Analytics, Speed Data** (à vérifier selon le contrat) | par segment et par période de 15 min (ou 5 min) : vitesse moyenne, vitesse de référence ou fluide, nombre de sondes ou confiance | variables de trafic de l'apprentissage |
| 1 | **Référentiel des segments** (table TMC ou géométrie des links HERE) | identifiant, coordonnées des points, longueur, sens, classe fonctionnelle | placer chaque segment sur l'axe (axe, sens, pk) |
| 2 | **Traffic Patterns** (vitesses typiques) | vitesse typique par segment, jour de la semaine et quart d'heure | référence « normale » et base de prévision à J+1 |
| 3 | Traffic Incidents (historique) | accidents, fermetures, travaux | variables d'événement, plus tard |

**Périmètre à demander.**

- **Réseau** : classe fonctionnelle 1 (autoroutes) sur les axes où nous avons des stations : A1, A4,
  A5, A6, A7, A8, A9, A10, A11, A13, A20, A26, A31, A36, A61, A62, A63, A71, A75, A89. Les deux sens.
- **Période** : la même que l'historique d'occupation, au moins 3 mois avec un début de vacances
  d'été. Idéalement mai à septembre 2026.
- **Pas de temps** : 15 min suffit, puisque l'occupation est agrégée à l'heure.
- **Format** : CSV ou Parquet. Le JSON par appel n'est pas adapté à un historique.

**Colonnes indispensables** dans l'historique (les noms exacts dépendent du produit) :

```
identifiant_segment, sens, horodatage_debut_periode, vitesse_moyenne, vitesse_fluide_ou_reference,
nombre_sondes (ou confiance), longueur_segment
```

### 2.2 Pour la production : le temps réel

L'**API Traffic Flow v7** donne, toutes les minutes environ, l'état de chaque segment. Elle sert
deux usages : l'estimation du moment et des 3 prochaines heures, et la constitution de notre propre
historique à partir de maintenant (section 5).

### 2.3 Règle de cohérence entre apprentissage et production

**Une variable n'entre dans le modèle que si elle existe aussi en production.** L'API temps réel ne
donne pas le nombre de sondes : on n'utilisera donc en direct que la vitesse, la vitesse fluide, le
jam factor et la confiance. Le nombre de sondes historique ne servira qu'aux variables décalées d'au
moins un jour, à condition que HERE le livre quotidiennement **(à vérifier)**.

## 3. Les variables qu'on en tire

Les segments HERE sont regroupés par tronçon de 10 km d'un même axe et d'un même sens, repéré par
son pk. Pour un tronçon $k$ composé de segments $i$ de longueur $L_i$ :

**Temps de parcours et vitesse du tronçon.** On moyenne les temps et non les vitesses, sinon un
court segment très lent serait sous-pondéré :

$$
T_k(t) = \sum_{i \in k} \frac{L_i}{v_i(t)},
\qquad
T^{\text{fluide}}_k = \sum_{i \in k} \frac{L_i}{v^{\text{fluide}}_i},
\qquad
v_k(t) = \frac{\sum_i L_i}{T_k(t)}
$$

**Indice de congestion** (0 = fluide, proche de 1 = bloqué) :

$$
c_k(t) = 1 - \frac{T^{\text{fluide}}_k}{T_k(t)} = 1 - \frac{v_k(t)}{v^{\text{fluide}}_k}
$$

**Jam factor et confiance** : moyennes pondérées par la longueur,
$J_k(t) = \sum_i L_i J_i(t) / \sum_i L_i$.

**Indice de volume par sondes** (historique). On le rapporte à la valeur habituelle du même tronçon,
le même jour de la semaine $j$ et à la même heure $h$, car la part de véhicules sondes varie d'un
tronçon à l'autre :

$$
\tilde n_k(t) = \frac{n_k(t)}{\operatorname{médiane}\left\{ n_k(t') : j(t') = j(t),\ h(t') = h(t) \right\}}
$$

Une valeur de 1,4 signifie 40 % de sondes de plus que d'habitude à cette heure-là.

**Écart à la normale.** Avec les Traffic Patterns ($v^{\text{typ}}_k$), l'anomalie de vitesse est
$a_k(t) = v_k(t) / v^{\text{typ}}_k(j, h) - 1$. C'est elle qui porte l'information « aujourd'hui
n'est pas un jour comme les autres ».

## 4. Comment le modèle les utilise

### 4.1 Les tronçons HERE deviennent des nœuds du graphe

Chaque tronçon est un nœud placé sur l'axe à son pk, dans son sens, sans délai de rattachement
puisqu'il est sur l'autoroute. Les stations et les compteurs Avatar restent en place. Le code de
`graphe.py` s'applique tel quel : un tronçon se traite comme un compteur de délai nul.

### 4.2 Le temps de parcours devient dynamique

Aujourd'hui, le temps entre deux nœuds est fixé à la vitesse de 110 km/h. Avec HERE, il devient le
temps mesuré :

$$
\tau_{ij}(t) = 60 \sum_{k \in \text{chemin}(i \to j)} T_k(t) \quad \text{(minutes)}
$$

Un bouchon entre un point amont et une aire retarde l'arrivée de la vague de véhicules ; le retard
$r_{cs}(t) = \operatorname{round}\left(\tau_{cs}(t)/\Delta\right)$ utilisé dans les variables de
propagation suit ce bouchon.

### 4.3 Variables pour le gradient boosting

Pour chaque station $s$ à l'instant $t$, avec $\mathcal K_s(a, b)$ les tronçons en amont situés
entre $a$ et $b$ km :

| Variable | Définition | Idée |
|---|---|---|
| congestion amont proche | moyenne de $c_k(t)$ sur $\mathcal K_s(0, 30)$ | bouchon juste avant l'aire |
| congestion amont lointaine | moyenne de $c_k(t - r_{ks})$ sur $\mathcal K_s(30, 200)$ | vague qui arrive |
| congestion maximale amont | $\max_k c_k$ sur $\mathcal K_s(0, 200)$ | événement ponctuel |
| anomalie de vitesse amont | moyenne de $a_k(t - r_{ks})$ sur $\mathcal K_s(0, 200)$ | jour inhabituel |
| volume par sondes propagé | $\sum_k \omega_{ks}\, \tilde n_k(t - r_{ks}) / \sum_k \omega_{ks}$, avec $\omega_{ks} = e^{-\tau_{ks}/90}$ | volume qui arrive, même fluide |
| congestion aval | moyenne de $c_k(t)$ sur les 30 km en aval | file qui remonte jusqu'à l'aire |
| temps depuis l'origine de l'axe | $\tau_{0s}(t)$ | autonomie consommée par les véhicules qui arrivent |

### 4.4 Pour le GNN

Les variables de chaque nœud tronçon sont $\left[c_k, J_k, a_k, \tilde n_k, \text{confiance}\right]$.
La matrice d'adjacence reste construite sur le temps fluide, pour rester stable. Le retard réel
entre dans le réseau par la fenêtre temporelle de 6 h que la convolution causale parcourt. Une
adjacence recalculée à chaque pas avec $\tau_{ij}(t)$ est possible, mais plus coûteuse : à garder
pour une seconde version.

### 4.5 Selon l'horizon

| Horizon | Données HERE utilisées | Remarque |
|---|---|---|
| Maintenant à 3 h | Traffic Flow temps réel | c'est là que HERE apporte le plus |
| J+1 à J+7 | Traffic Patterns + calendrier + couleur Bison Futé | la congestion de demain n'est pas observée : l'étape 1 la prévoit |

À J+1, la congestion observée aujourd'hui sert surtout à recaler la prévision (un jour de grève,
de météo dégradée, ou une fermeture longue).

## 5. Connexion à l'API Traffic Flow v7

### 5.1 Requête

```
GET https://mobicloud.ifpen.com/geoservices/api/here/data.traffic.hereapi.com/v7/flow
    ?in=bbox:{ouest},{sud},{est},{nord}
    &locationReferencing=shape
Authorization: Bearer {token}
```

- **Authentification** : pas de clé HERE directe. Le script passe par le proxy Mobicloud de l'IFPEN
  (`~/.mobicloud/login.json`, comme `here_functions.py`), qui expose l'API HERE sous
  `mobicloud.ifpen.com/geoservices/api/here/{hôte HERE}/{chemin}` avec un jeton Bearer obtenu via
  `get_token()`. Vérifié en appelant réellement l'API le 19 octobre 2026 (axe A10) : la réponse a la
  même structure que l'API publique.
- **Zone** : `in=bbox:...` ou `in=circle:{lat},{lon};r={mètres}`. La taille maximale d'une bbox est
  limitée **(à vérifier)** : le script découpe chaque axe en boîtes de 60 km.
- **Référencement** : `shape` renvoie les coordonnées de chaque segment, ce qui permet de les
  projeter sur nos axes. `tmc` renvoie les codes TMC, utile si l'historique est livré en TMC.

### 5.2 Réponse (structure simplifiée)

```json
{
  "sourceUpdated": "2026-10-17T12:00:00Z",
  "results": [
    {
      "location": {
        "description": "A10",
        "length": 2000,
        "shape": { "links": [ { "points": [ {"lat": 47.59, "lng": 1.33}, {"lat": 47.58, "lng": 1.31} ] } ] }
      },
      "currentFlow": {
        "speed": 33.1, "speedUncapped": 34.2, "freeFlow": 36.1,
        "jamFactor": 0.8, "confidence": 0.95, "traversability": "open"
      }
    }
  ]
}
```

Les vitesses sont en **m/s** (multiplier par 3,6). `speedUncapped` est la vitesse réellement
mesurée ; `speed` est plafonnée à la vitesse fluide. On utilise `speedUncapped` quand il est présent.

### 5.3 Le script `here_flux.py`

Pour chaque axe demandé, le script :

1. découpe le tracé (`axes_approx.csv`) en boîtes de 60 km et interroge l'API, avec réessai en cas
   d'erreur 429 (quota dépassé) ;
2. garde les segments **autoroutiers qui suivent l'axe** : à moins de 3 km du tracé, vitesse fluide
   d'au moins 90 km/h, orientation à moins d'environ 37° de celle de l'axe ($\lvert\cos\rvert \ge 0{,}8$) ;
3. leur attribue un pk par projection sur le tracé, et un sens selon leur orientation
   (sens 1 = vers les pk croissants, même convention que les stations) ;
4. agrège par tronçon de 10 km (formules de la section 3) et ajoute une ligne par tronçon à
   `here_troncons.csv`.

```bash
# Windows, avec le venv du projet et ~/.mobicloud/login.json déjà renseigné
.\.venv\Scripts\python.exe here_flux.py --axes donnees_reelles\axes_approx.csv --axe A10 --axe A6 --axe A7 --sortie donnees_reelles\here
```

Colonnes de `here_troncons.csv` :

```
horodatage, axe, sens, troncon_pk, longueur_couverte_km, nb_segments,
vitesse_kmh, vitesse_fluide_kmh, congestion, temps_parcours_min, jam_factor, confiance
```

**État du script.** D'abord testé sur une réponse simulée (A10 avec un bouchon entre les pk 100 et
130 : congestion retrouvée à 0,67, deux sens bien séparés), puis **sur l'API réelle le 19 octobre
2026** via le proxy Mobicloud, sur l'axe A10 : 6345 segments reçus, 138 retenus sur l'autoroute,
76 tronçons, congestion moyenne cohérente avec un trafic fluide à l'heure de l'appel. Le filtre
d'orientation reste à surveiller sur les axes obliques (une route perpendiculaire rapide peut le
passer) : vérifier les segments retenus avec `--garder-brut` avant de planifier un axe pour la
première fois.

### 5.4 Fréquence et volume d'appels

| Fréquence | Appels par passage (20 axes) | Appels par jour |
|---|---|---|
| 5 min | environ 120 | environ 35 000 |
| 15 min | environ 120 | environ 11 500 |

Le pas de 15 min suffit pour un modèle horaire. Le quota de votre contrat est **à vérifier** avant
de planifier la tâche.

**Planification sous Windows** (toutes les 15 min) :

```bat
schtasks /Create /SC MINUTE /MO 15 /TN "HERE_flux_IRVE" ^
  /TR "cmd /c cd /d C:\chemin\vers\irve && python here_flux.py --axes donnees_reelles\axes_approx.csv --axe A10 --axe A6 --axe A7 --sortie donnees_reelles\here"
```

La clé doit alors être une variable d'environnement de l'utilisateur (`setx HERE_API_KEY ...`),
pas écrite dans la commande.

## 6. Questions à poser à HERE ou à vérifier dans le contrat

1. Quel produit historique est inclus : Traffic Analytics Speed Data, Traffic Patterns, autre ?
2. Le nombre de sondes ou la taille d'échantillon est-il fourni par segment et par période ?
3. Référencement de l'historique : TMC, links HERE, OpenLR ? Avec la table de géométrie ?
4. Quotas de l'API Traffic Flow (appels par seconde et par mois) et taille maximale d'une bbox ?
5. Peut-on filtrer l'API temps réel par classe fonctionnelle, pour ne recevoir que les autoroutes ?

## 7. Suite

1. Récupérer l'extrait d'historique et le référentiel des segments (section 2.1).
2. Placer les segments sur les axes avec la même projection que les stations, et vérifier sur un axe.
3. Calculer les variables de la section 4.3 et mesurer, comme pour Avatar, leur corrélation avec
   l'occupation hors cycle journalier.
4. Les ajouter à l'étape 2 et comparer avec le modèle sans trafic et avec Avatar seul.
5. Lancer la collecte temps réel dès que la clé est disponible, pour constituer notre propre historique.
