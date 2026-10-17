# Prévision de l'occupation des bornes de recharge sur autoroute : note technique

*Projet « Occupation IRVE », état au 17 octobre 2026. Code : dépôt `gillescorde/occupation-irve`.
Note complémentaire sur les données HERE : `note_here.md`.*

## 1. Objet et état d'avancement

L'objectif est un « Bison Futé » de l'occupation des stations de recharge rapide sur les grands axes.
On prévoit le taux d'occupation de chaque station à un horizon $H$ (24 h par défaut), puis on le
restitue sous forme de couleur. La chaîne retenue a deux étapes : prévoir le trafic en quelques points
de comptage, puis en déduire l'occupation des stations en tenant compte de la propagation du trafic
le long du réseau.

| Brique | Fichier | État |
|---|---|---|
| Conversion de l'historique Qualicharge | `donnees.py` | implémentée, appliquée aux données réelles |
| Placement des stations sur les axes | `reperer_stations_autoroute.py`, `preparer_stations.py` | implémentée, appliquée au référentiel réel |
| Graphe routier et variables de propagation | `graphe.py` | implémentée, testée sur données synthétiques |
| Rattachement des compteurs Avatar au graphe | `rattacher_compteurs.py` | implémentée, appliquée aux débits réels (mai à juillet 2026) |
| Client HERE Traffic Flow | `here_flux.py` | implémenté, testé sur une réponse simulée uniquement |
| Étape 1 : prévision des débits | `entrainer.py` | implémentée, testée sur données synthétiques |
| Étape 2 : gradient boosting et GNN spatio-temporel | `entrainer.py`, `gnn.py` | implémentée, testée sur données synthétiques |
| Générateur de données synthétiques | `generer_donnees_synthetiques.py` | implémenté |
| Scénarios de pénétration VE | `scenarios.py` | implémentée, appliquée à l'occupation réelle |
| Restitution web (Flask) | | à faire |

**Point important : la chaîne complète n'a pas encore été apprise sur données réelles.** Elle a été
validée sur un jeu synthétique qui imite la situation réelle (section 6). Sur les données réelles, on
dispose de l'occupation du 9 au 15 juillet 2026 seulement et des débits Avatar de mai à juillet
2026, rattachés au graphe (section 2.3). L'apprentissage réel attend l'historique long d'occupation
et les données HERE (section 9).

## 2. Données

### 2.1 Historique d'occupation (Qualicharge)

Source : collecte de l'API `map.qualicharge.beta.gouv.fr/api/irve/points/`, toutes les 5 minutes.
Le fichier est un **journal de changements d'état** : une ligne $e$ par station et par nouvel état,
avec :

- $a_e$ : `fetched_at`, première collecte de cet état ;
- $\ell_e$ : `last_seen_at`, dernière collecte où l'état a été revu ;
- $\sigma_e$ : `latest_status_at`, horodatage transmis par l'opérateur ;
- les effectifs de points de charge (PDC) : $n^{\text{libre}}_e$, $n^{\text{occ}}_e$, $n^{\text{rés}}_e$,
  $n^{\text{inc}}_e$ (inconnu), $n^{\text{serv}}_e$ (en service), $n^{\text{tot}}_e$.

On vérifie sur les données que $n^{\text{libre}} + n^{\text{occ}} + n^{\text{rés}} + n^{\text{inc}} = n^{\text{tot}}$
sur 100 % des lignes.

**Intervalle de validité d'un état.** L'état $e$ est valable sur $[a_e, b_e)$ avec

$$
b_e = \min\left(a_{e+1},\; \ell_e + 5\,\text{min},\; \sigma_e + 24\,\text{h}\right)
$$

où $a_{e+1}$ est l'état suivant de la même station. Le troisième terme écarte les remontées mortes :
un statut que l'opérateur n'a pas rafraîchi depuis plus de 24 h devient inconnu. Le deuxième terme
fait que les trous de collecte deviennent des valeurs manquantes au lieu d'être comblés.

**Agrégation sur un pas de temps** $[t, t+\Delta)$, avec $\Delta = 1$ h. Pour chaque état, le poids
est la fraction du pas qu'il recouvre :

$$
w_e(t) = \frac{\left|\,[a_e, b_e) \cap [t, t+\Delta)\,\right|}{\Delta},
\qquad
\bar n_x(s,t) = \frac{\sum_e w_e(t)\, n_{x,e}}{\sum_e w_e(t)},
\qquad
c(s,t) = \sum_e w_e(t)
$$

Un pas dont la couverture $c(s,t)$ est inférieure à 0,5 est abandonné.

**Taux d'occupation.** Les PDC à l'état inconnu sont exclus du dénominateur au lieu d'être comptés
libres, et les PDC réservés sont comptés occupés :

$$
y(s,t) = \frac{\bar n^{\text{occ}} + \bar n^{\text{rés}}}{\bar n^{\text{libre}} + \bar n^{\text{occ}} + \bar n^{\text{rés}}} \in [0, 1]
$$

**Données réelles disponibles.** Le fichier complet (350 Mo, 1,28 million de lignes, 7 388
stations) couvre seulement du 9 au 15 juillet 2026, avec deux trous de collecte (9 h le 9 juillet,
3 h 30 le 10 juillet). C'est insuffisant pour l'apprentissage : il faut au moins 2 à 3 mois pour
voir plusieurs week-ends et un début de vacances. Un historique plus long est attendu.

### 2.2 Référentiel des stations et placement sur les axes

Le référentiel Qualicharge (7 388 stations, toutes de 50 kW ou plus) donne les coordonnées, le nombre
de PDC, la puissance et l'opérateur, mais pas l'autoroute. Chaque station reçoit un axe, un sens et
un point kilométrique $\text{pk}$.

**Axe.** Il est lu dans le nom quand il y figure (« Electra A10 - Aire de Cézac »). Sinon, pour un
nom d'aire, on prend l'axe le plus proche à moins de 5 km. Les bornes réservées aux poids lourds
sont écartées.

**pk.** Faute de tracé exact (OSM et data.gouv ne sont pas accessibles depuis l'environnement de
calcul), chaque axe est approché par une ligne brisée passant par des villes étapes, densifiée tous
les kilomètres. Le pk d'un point du tracé est la distance cumulée (haversine) depuis la première
ville. La station est projetée orthogonalement sur le segment le plus proche :

$$
\theta = \operatorname{clip}\!\left(\frac{(\mathbf p - \mathbf a)\cdot(\mathbf b - \mathbf a)}{\lVert \mathbf b - \mathbf a\rVert^2},\, 0,\, 1\right),
\qquad
\text{pk} = \text{pk}_a + \theta\,(\text{pk}_b - \text{pk}_a)
$$

dans une projection équirectangulaire locale. L'erreur est de quelques kilomètres, acceptable pour
des aires espacées de 20 à 60 km.

**Sens.** Convention : sens 1 = sens des pk croissants, sens 2 = sens inverse, 0 = inconnu.

- Si le nom indique une destination (« direction Paris », « Sens Bordeaux Paris ») :
  $\text{sens} = 1$ si $\text{pk}(\text{destination}) > \text{pk}(s)$, sinon 2.
- Sinon, si le nom porte un suffixe cardinal (Nord, Sud, Est, Ouest), on utilise le côté de la
  chaussée. On roule à droite : soit $\mathbf u = (u_x, u_y)$ le vecteur unitaire du tracé dans le
  sens des pk croissants (est, nord), $\mathbf n = (u_y, -u_x)$ sa normale à droite et $\mathbf v$
  le vecteur cardinal. Alors $\text{sens} = 1$ si $\mathbf v \cdot \mathbf n > 0$, sinon 2.
- Sur toutes les stations qui portent à la fois une destination et un suffixe, les deux règles
  donnent le même sens.

**Résultat.** 170 stations d'autoroute sur 20 axes : 119 avec l'axe lu dans le nom et 51 déduites de
la proximité. Le sens vient de la destination pour 38 d'entre elles, du côté de la chaussée pour 47,
et reste inconnu pour 85. Dans l'historique de juillet, 150 de ces stations ont des données. Leur
occupation moyenne horaire culmine à 37 % vers 13 h, contre 22 % pour les autres stations.

**Limite.** Une station d'aire nommée seulement par sa commune ou son opérateur n'est pas détectée.
`preparer_stations.py` fait la même projection sur un vrai tracé d'autoroute fourni en entrée
(axe, pk, lat, lon) et retient toute station à moins de 300 m, sans passer par le nom.

### 2.3 Débits routiers (compteurs Avatar)

Format attendu : `comptages.csv` (horodatage, compteur_id, débit en véh/h) et `compteurs.csv`
(compteur_id, coordonnées, axe de rattachement, sens, pk projeté, délai de rattachement en minutes).
Les compteurs (type Avatar) ne sont pas sur les autoroutes : un compteur sur une route parallèle ou
une pénétrante est **rattaché** à l'axe au pk de sa projection, avec un délai $\delta_c$ qui
représente le trajet jusqu'à l'autoroute.

Comme un tel compteur ne mesure pas le flux autoroutier mais un indicateur corrélé, son débit est
normalisé par sa moyenne sur la période d'apprentissage :

$$
\tilde q_c(t) = \frac{q_c(t)}{\bar q_c},
\qquad
\bar q_c = \frac{1}{|\mathcal T_{\text{app}}|}\sum_{t \in \mathcal T_{\text{app}}} q_c(t)
$$

Le modèle exploite ainsi les **variations** du trafic (1 = trafic habituel du compteur), pas son niveau.

**Données réelles.** 1 184 compteurs Avatar (réseau non concédé, à moins de 5 km des axes), débits
horaires du 1er mai au 31 juillet 2026. Les heures estimées à plus de 50 % par Avatar sont écartées
(colonne `part_estimee_pct`). Les compteurs se concentrent aux extrémités urbaines des autoroutes ;
l'A9 est inexploitable (4 compteurs estimés à plus de 80 %).

**Rattachement et premier test** (`rattacher_compteurs.py`, semaine du 9 au 15 juillet) : on mesure
la corrélation entre le trafic amont propagé $x^{(0)}_s$ et l'occupation, brute puis après avoir
retiré à chaque série sa moyenne par heure de la journée (« hors cycle »), qui seule mesure
l'information apportée en plus de l'heure.

| Compteurs utilisés | Stations reliées | $r$ brute (médiane) | $r$ hors cycle (médiane) |
|---|---|---|---|
| Tous (997) | 113 / 149 | 0,55 | 0,07 |
| Interurbains seulement (96) | 71 / 149 | 0,58 | 0,17 |

Les compteurs urbains diluent le signal : seuls les compteurs interurbains sont à retenir. Sur
environ 140 heures par station, ces chiffres restent fragiles.

### 2.4 Calendrier

Variables par jour : jour férié, vacances scolaires, deux premiers et deux derniers jours de
vacances, et la couleur Bison Futé officielle (0 à 3) quand elle est disponible. L'heure et le jour
de la semaine sont encodés en sinus et cosinus :
$\left(\sin\frac{2\pi h}{24}, \cos\frac{2\pi h}{24}, \sin\frac{2\pi j}{7}, \cos\frac{2\pi j}{7}\right)$.

## 3. Le graphe routier

### 3.1 Nœuds et arêtes

Les nœuds sont les $S$ stations puis les $C$ compteurs, soit $N = S + C$. Les arêtes sont orientées
dans le sens de circulation :

- sur chaque couple (axe, sens), chaque station est reliée à la suivante, les stations étant
  triées par pk croissant en sens 1 et décroissant en sens 2 ;
- chaque compteur est relié à la première station en aval de son point de rattachement ;
- une station de sens inconnu est insérée dans les deux sens.

Le poids d'une arête est un temps de parcours en minutes, avec $v = 110$ km/h :

$$
\tau_{ij} = \frac{60\,\lvert \text{pk}_j - \text{pk}_i \rvert}{v} + \delta_i
$$

où $\delta_i$ est le délai de rattachement si $i$ est un compteur, et 0 sinon. Les plus courts
chemins (Dijkstra, graphe orienté) donnent la matrice $\mathbf T \in \mathbb R^{N \times N}$ des
temps de parcours de tout nœud vers tout nœud en aval, avec $T_{ij} = +\infty$ si $j$ n'est pas en
aval de $i$.

Pour un réseau maillé (bifurcations A10/A11, A6/A31...), la construction automatique par (axe, pk)
doit être remplacée par un fichier `aretes.csv` tiré d'un vrai graphe routier. Le reste du code
n'en dépend pas.

### 3.2 Matrices d'adjacence

Comme pour DCRNN, on applique un noyau gaussien au temps de parcours, avec $\sigma = 60$ min et
$\tau_{\max} = 240$ min :

$$
W_{ij} = \exp\!\left(-\left(\frac{T_{ij}}{\sigma}\right)^{2}\right)\mathbf 1\!\left[T_{ij} \le \tau_{\max}\right],
\qquad W_{ii} = 0
$$

On en tire deux matrices normalisées par ligne :

$$
\mathbf A^{\uparrow} = \mathbf D_{\uparrow}^{-1}\mathbf W^{\top},
\qquad
\mathbf A^{\downarrow} = \mathbf D_{\downarrow}^{-1}\mathbf W
$$

La ligne $j$ de $\mathbf A^{\uparrow}$ (amont) pondère les nœuds dont le trafic arrive en $j$. La
matrice $\mathbf A^{\downarrow}$ (aval) porte l'effet inverse : une aire saturée en aval peut
reporter la demande vers l'amont. Un nœud peut ainsi recevoir directement l'information d'un nœud
situé plusieurs arêtes en amont, s'il est à moins de 4 h.

## 4. Chaîne de prévision

### 4.1 Contrainte d'horizon

On prévoit $y(s, t)$ à l'instant $t_0 = t - H$. Une occupation passée n'est utilisable que si elle
date d'au moins $H$. Avec un pas horaire, les retards autorisés sont :

$$
\mathcal L(H) = \{\, k \in \{H,\ 24,\ 48,\ 168,\ 336\} \;:\; k \ge H \,\}
$$

Pour $H = 24$ h, on a $\mathcal L = \{24, 48, 168, 336\}$. On utilise aussi une moyenne glissante sur
24 h décalée de $H$ :

$$
\bar y^{24}(s,t) = \frac{1}{24}\sum_{i=0}^{23} y(s, t - H - i)
$$

### 4.2 Étape 1 : prévision des débits aux compteurs

Un seul modèle de gradient boosting (`HistGradientBoostingRegressor` de scikit-learn : 400
itérations, pas 0,05, arrêt anticipé) est appris pour tous les compteurs. L'identifiant du compteur
est une variable catégorielle. Les variables sont le calendrier, $\tilde q_c(t-k)$ pour
$k \in \mathcal L(H)$ et la moyenne sur 24 h de $\tilde q_c$ décalée de $H$. La cible est
$\tilde q_c(t)$. La prévision $\hat{\tilde q}_c(t)$ alimente l'étape 2 en production. Les trous de
mesure du débit observé sont comblés par cette prévision.

### 4.3 Variables de propagation (pour le gradient boosting)

La propagation est encodée explicitement. Soit $\tau_{cs} = T_{cs}$ le temps de parcours du compteur
$c$ à la station $s$, et $r_{cs} = \operatorname{round}(\tau_{cs}/\Delta)$ le retard en pas de
temps. Les compteurs retenus sont ceux en amont à moins de $\tau_{\max} = 300$ min. Leur poids
décroît avec la distance, avec $\tau_0 = 90$ min :

$$
\omega_{cs} = \exp\!\left(-\frac{\tau_{cs}}{\tau_0}\right)
$$

Le trafic amont propagé, avec un décalage supplémentaire $k \in \{0, 1, 2, 3\}$ h pour représenter
l'inertie des files d'attente, vaut :

$$
x^{(k)}_s(t) = \frac{\sum_{c} \omega_{cs}\, \tilde q_c\!\left(t - r_{cs} - k\right)}{\sum_{c} \omega_{cs}\, \mathbf 1\!\left[\tilde q_c(t - r_{cs} - k)\ \text{observé}\right]}
$$

On ajoute le compteur amont le plus proche seul, $\tilde q_{c^*}(t - r_{c^* s})$ avec
$c^* = \arg\min_c \tau_{cs}$. Le retard $r_{cs}$ décale chaque compteur d'une durée différente : une
vague de départs vue à la sortie de Paris est ainsi attribuée à une aire située à 300 km environ
2 h 45 plus tard.

Pour les stations voisines, on prend l'occupation connue il y a $H$, pondérée par le graphe et
restreinte aux stations observées :

$$
v^{\uparrow}_s(t) = \frac{\sum_j A^{\uparrow}_{sj}\, y(j, t - H)}{\sum_j A^{\uparrow}_{sj}\, \mathbf 1[y(j, t-H)\ \text{observé}]}
$$

et de même $v^{\downarrow}_s(t)$ avec $\mathbf A^{\downarrow}$.

### 4.4 Étape 2a : gradient boosting

Une ligne correspond à un couple (station, heure). Les variables sont :

- le calendrier ;
- des variables statiques : identifiant de la station (catégoriel tant qu'il y a au plus 255
  stations), nombre de PDC, pk, puissance et sens ;
- les retards d'occupation $y(s, t-k)$ pour $k \in \mathcal L(H)$, et $\bar y^{24}$ ;
- l'occupation des voisines $v^{\uparrow}_s$ et $v^{\downarrow}_s$ ;
- le trafic propagé $x^{(0..3)}_s$ et le compteur amont le plus proche.

Le modèle est un `HistGradientBoostingRegressor` (600 itérations, pas 0,05, 31 feuilles,
régularisation $L_2$ = 1, arrêt anticipé), dont la sortie est bornée à $[0, 1]$. Les valeurs
manquantes sont gérées nativement, ce qui couvre les stations sans compteur amont.

### 4.5 Étape 2b : GNN spatio-temporel (`gnn.py`)

L'architecture s'inspire de STGCN et de Graph WaveNet. Ici, la propagation est apprise au lieu
d'être calculée à la main.

**Entrée.** Le tenseur est $\mathbf X \in \mathbb R^{B \times T_{\text{in}} \times N \times F}$ :
pour chaque nœud, une fenêtre de $T_{\text{in}} = 6$ pas se terminant à $t$. Les variables sont le
débit normalisé (nul pour les stations), un indicateur de type de nœud, les occupations
$y(s, t'-k)$ pour $k \in \mathcal L(H)$, $\bar y^{24}$ et un masque indiquant si $y(s, t'-H)$ est
observé (nuls pour les compteurs), le calendrier en sinus/cosinus diffusé à tous les nœuds, et la capacité
$n^{\text{tot}}/10$. On concatène à chaque nœud un plongement appris
$\mathbf p_n \in \mathbb R^{8}$, puis une couche linéaire ramène l'ensemble à $d = 32$ canaux :
$\mathbf H^{(0)} = \mathbf W_{\text{in}}[\mathbf X \,\|\, \mathbf p]$.

**Convolution temporelle causale à porte.** Le noyau de taille 3 est appliqué le long du temps, nœud
par nœud, en ne regardant que le passé :

$$
\Gamma(\mathbf H) = \tanh\!\left(\Theta_1 \star \mathbf H\right) \odot \operatorname{sigmoïde}\!\left(\Theta_2 \star \mathbf H\right)
$$

**Convolution de graphe par diffusion.** On utilise trois supports : l'amont, l'aval et une matrice
adaptative apprise, comme dans Graph WaveNet :

$$
\mathbf A^{\text{adp}} = \operatorname{softmax}\!\left(\operatorname{ReLU}\!\left(\mathbf E_1 \mathbf E_2^{\top}\right)\right),
\qquad \mathbf E_1, \mathbf E_2 \in \mathbb R^{N \times 8}
$$

$$
\mathcal G(\mathbf H) = \mathbf W_0 \mathbf H + \sum_{\mathbf A \in \{\mathbf A^{\uparrow},\, \mathbf A^{\downarrow},\, \mathbf A^{\text{adp}}\}} \;\sum_{k=1}^{K} \mathbf W_{\mathbf A, k}\, \mathbf A^{k} \mathbf H,
\qquad K = 2
$$

Le produit $\mathbf A\mathbf H$ est la moyenne pondérée des voisins, et $\mathbf A^2\mathbf H$
celle des voisins de voisins. Les poids $\mathbf W$ sont partagés par tous les nœuds. La matrice
$\mathbf A^{\text{adp}}$ corrige ce que le graphe théorique ignore : compteur mal rattaché, report
de clientèle vers une station hors autoroute, etc.

**Bloc résiduel**, répété $L = 2$ fois, avec un dropout de 0,1 :

$$
\mathbf H^{(l+1)} = \operatorname{BatchNorm}\!\left(\operatorname{Dropout}\!\left(\mathcal G\!\left(\Gamma\!\left(\mathbf H^{(l)}\right)\right)\right) + \mathbf H^{(l)}\right)
$$

**Sortie.** On lit le dernier pas de temps pour chaque station :

$$
\hat y(s, t) = \operatorname{sigmoïde}\!\left(\mathbf w_2^{\top}\operatorname{ReLU}\!\left(\mathbf W_1 \mathbf h^{(L)}_{s,t} + \mathbf b_1\right) + b_2\right)
$$

**Apprentissage.** La perte est l'erreur absolue moyenne sur les stations observées, avec
$m_{s,t} = 1$ si $y(s,t)$ est connu :

$$
\mathcal L = \frac{\sum_{t}\sum_{s=1}^{S} m_{s,t}\,\lvert \hat y(s,t) - y(s,t)\rvert}{\sum_{t}\sum_{s} m_{s,t}}
$$

L'optimiseur est AdamW (pas $2\cdot10^{-3}$, décroissance des poids $10^{-4}$), avec des lots de 64,
un écrêtage du gradient à 5, au plus 30 époques et un arrêt anticipé après 6 époques sans
amélioration de la MAE de validation.

## 5. Protocole d'évaluation

**Découpage temporel** : 70 % pour l'apprentissage, 15 % pour la validation et 15 % pour le test,
dans l'ordre chronologique. Un découpage aléatoire laisserait fuir l'information des heures voisines.

**Double évaluation de l'étape 2** sur la période de test :

- *trafic observé* : les variables de trafic utilisent $\tilde q_c$. C'est une borne haute, comme si
  l'étape 1 était parfaite ;
- *trafic prévu* : elles utilisent $\hat{\tilde q}_c$ de l'étape 1. C'est la chaîne complète, telle
  qu'elle tournera en production.

**Références** : le profil moyen par station, jour de la semaine et heure, et un gradient boosting
sans aucune variable de trafic. Cette seconde référence mesure ce que le trafic apporte réellement.

**Couleurs** :

$$
\text{couleur}(y) =
\begin{cases}
\text{vert} & y < 0{,}40 \\
\text{orange} & 0{,}40 \le y < 0{,}70 \\
\text{rouge} & 0{,}70 \le y < 0{,}90 \\
\text{noir} & y \ge 0{,}90
\end{cases}
$$

**Métriques** : la MAE et la RMSE en points de taux, la part de couleurs exactes, le rappel des
alertes (part des heures réellement rouges ou noires prévues rouges ou noires) et leur précision.

## 6. Validation sur données synthétiques

### 6.1 Générateur

Le scénario comporte deux axes (A10 sur 560 km et A6 sur 460 km), dans les deux sens, du 1er mars au
31 août 2026 au pas horaire, avec 32 stations et 12 compteurs.

- **Demande à l'origine** $D(t)$ : deux pointes journalières (9 h et 17 h 30), puis des modulations
  pour les vacances, l'été, le vendredi après-midi, le samedi des vacances à l'aller, le dimanche au
  retour, et les débuts et fins de vacances. On y ajoute un bruit horaire lissé et un aléa journalier
  log-normal d'écart-type 0,18 (météo, événement). Cet aléa est vu par les compteurs mais
  imprévisible par le calendrier.
- **Flux autoroutier caché** au pk $x$, à la distance $\xi$ de l'origine du sens :
  $F(x,t) = D\!\left(t - \xi/v\right)\,(1 - 0{,}0006\,\xi) + \varepsilon$.
- **Compteurs hors autoroute** :
  $q_c(t) = \alpha_c F(x_c,t) + \ell(t) + \varepsilon$, avec $\alpha_c \sim \mathcal U(0{,}15;\ 0{,}4)$,
  une composante locale $\ell$ aux heures de pointe et 2 % de valeurs manquantes.
- **Occupation** :
  $\lambda_s(t) = 0{,}08\,F(x_s,t)\,\pi(t)\,\beta(\xi_s)$, où $\pi(t)$ est la part de véhicules
  électriques (3,5 %, en légère hausse) et $\beta(\xi) = \operatorname{clip}\!\left(\frac{\xi - 80}{200}, 0{,}05, 1\right)$
  traduit qu'on recharge rarement juste après le départ. Le taux attendu est
  $\rho_s = 1 - e^{-1{,}3\,\lambda_s/n_s}$, et l'occupation observée est la moyenne de 4 tirages
  binomiaux $\mathcal B(n_s - n^{\text{hs}}_s, \rho_s)$, avec des pannes aléatoires.

### 6.2 Résultats (horizon 24 h, test sur août)

| Modèle | MAE (points) | RMSE (points) | Couleur exacte | Rappel alertes | Précision alertes |
|---|---|---|---|---|---|
| Profil moyen station × jour × heure | 10,4 | 13,7 | 73 % | 18 % | 96 % |
| Gradient boosting sans trafic | 7,3 | 9,5 | 82 % | 64 % | 75 % |
| Gradient boosting, trafic observé | 6,2 | 8,2 | 85 % | 67 % | 84 % |
| Gradient boosting, trafic prévu | 7,3 | 9,7 | 81 % | 73 % | 68 % |
| GNN, trafic observé | 6,1 | 8,1 | 85 % | 72 % | 79 % |
| GNN, trafic prévu | 7,4 | 9,9 | 81 % | 80 % | 63 % |

### 6.3 Interprétation

1. **Le trafic apporte de l'information quand il est connu** : 6,2 points contre 7,3 sans trafic.
   C'est le cas de l'estimation en temps réel ou à très court terme (1 à 3 h) à partir des compteurs.
2. **À J+1, ce gain disparaît si l'étape 1 ne s'appuie que sur le calendrier**, information que
   l'étape 2 possède déjà. La chaîne en deux étapes n'a d'intérêt à J+1 que si la prévision de
   trafic apporte autre chose : prévision Bison Futé officielle, événements, météo.
3. **GNN et gradient boosting font jeu égal** sur ce réseau de 44 nœuds. Le GNN détecte un peu mieux
   les saturations, au prix de plus de fausses alertes. Avec peu d'historique et quelques dizaines de
   stations, le gradient boosting avec variables de graphe est recommandé en premier : il est plus
   simple, explicable et robuste. Le GNN deviendra intéressant avec un réseau plus dense et un
   historique plus long.
4. Ces résultats viennent de données synthétiques et ne préjugent pas des performances réelles.

## 7. Scénarios de pénétration des véhicules électriques (`scenarios.py`)

**Objectif.** Simuler l'occupation des aires pour une part de VE dans le trafic $p'$ (par exemple
10 % en 2030) différente de celle de l'historique, $p_{\text{ref}} \approx 3\,\%$.

**Pourquoi pas une simple variable d'entrée.** Sur l'historique, $p$ est quasi constant. Un modèle
appris ne peut rien tirer d'une variable sans variation, et un gradient boosting n'extrapole pas
au-delà des valeurs vues. La pénétration entre donc dans une **structure physique** et non dans
l'apprentissage.

**Charge offerte.** On note $A(s,t)$ le nombre moyen de véhicules qui voudraient être en charge en
même temps à la station $s$ (en erlangs). Elle est proportionnelle au flux de VE qui passe :

$$
A(s,t) = p \cdot Q_s(t) \cdot \beta_s(t) \cdot \bar D
$$

avec $Q_s$ le flux total, $\beta_s$ la part des VE qui s'arrêtent recharger et $\bar D$ la durée
moyenne de charge. Pour un scénario, seuls $p$ et éventuellement $\beta$ changent :

$$
A'(s,t) = A(s,t)\,\frac{p'}{p_{\text{ref}}}\,\kappa
$$

où $\kappa$ (`--facteur-besoin`) traduit l'évolution du besoin de recharge par VE (moins de 1 si les
batteries de 2030 sont plus grandes).

**De la charge à l'occupation : Erlang B.** Avec $c$ points de charge, la probabilité que tous soient
occupés et le taux d'occupation moyen valent :

$$
B(A, c) = \frac{A^c / c!}{\sum_{k=0}^{c} A^k / k!},
\qquad
y = \frac{A\,\bigl(1 - B(A, c)\bigr)}{c}
$$

$B$ se calcule par la récurrence stable $B_0 = 1$, $B_k = \frac{A B_{k-1}}{k + A B_{k-1}}$.
La relation $y(A)$ est croissante : on l'inverse par dichotomie pour retrouver $A$ à partir d'une
occupation observée ou prévue, on applique le scénario, puis on recalcule $y'$ et $B'$. L'occupation
sature naturellement à 100 %, et $B'$ donne directement la **probabilité de trouver la station pleine**,
l'indicateur le plus parlant pour un « Bison Futé ». Le nombre de PDC peut aussi être modifié
(`--pdc-supplementaires`, `--facteur-pdc`) pour tester un plan d'équipement.

**Limite.** Au-delà de 97 % d'occupation, la charge réelle n'est plus observable (les véhicules
refusés ne sont pas comptés) : elle est plafonnée, ce qui sous-estime les scénarios des stations déjà
saturées. Erlang B suppose aussi que les véhicules qui trouvent la station pleine repartent ; s'ils
attendent, un modèle M/M/c avec file (Erlang C) donnerait des temps d'attente.

**Premiers résultats sur l'occupation réelle (150 stations d'autoroute, 9 au 15 juillet 2026).**

| Scénario | Occupation moyenne | Heures rouge ou noir | Probabilité moyenne de station pleine |
|---|---|---|---|
| 3 % (référence) | 18,5 % | 3,5 % | 2,1 % |
| 6 % | 31,2 % | 15,1 % | 6,1 % |
| 10 % | 42,2 % | 29,3 % | 12,4 % |
| 15 % | 50,7 % | 41,1 % | 19,3 % |
| 10 %, +4 PDC par station | 36,1 % | 21,6 % | 6,9 % |
| 10 %, besoin par VE × 0,7 | 34,5 % | 19,2 % | 7,7 % |

**Évolution prévue.** Quand l'historique long sera disponible, l'étape 2 sera apprise directement
sur la charge offerte $A$ plutôt que sur $y$. Le modèle prédira alors une grandeur qui se met à
l'échelle avec $p$, et la conversion par Erlang B se fera en sortie.

## 8. La prévision en production

### 8.1 Deux horizons, deux sources de trafic

L'API HERE Traffic Flow donne l'état **actuel** du trafic, pas une prévision. Les entrées dépendent
donc de l'horizon :

| Horizon | Trafic utilisé | Autres entrées |
|---|---|---|
| Maintenant à +3 h | HERE temps réel sur les tronçons en amont de chaque aire, propagé avec le temps de parcours mesuré (`note_here.md`, section 4) ; compteurs Avatar interurbains | occupation des dernières heures, calendrier |
| J+1 à J+7 | prévision de l'étape 1 : vitesses typiques HERE (Traffic Patterns), calendrier, couleur Bison Futé, état des derniers jours | occupation décalée d'au moins $H$, calendrier |
| Au-delà de J+7 (semaines, mois) | prévision de l'étape 1 sans les derniers jours : calendrier, couleur Bison Futé, Traffic Patterns | calendrier ; restitution en médiane et P90 (section 8.3) |

À J+1, la couleur Bison Futé n'est qu'une entrée parmi d'autres.

**Entrées du simulateur une fois appris.** Le graphe, les temps de parcours et la réponse de chaque
station au trafic sont figés dans les paramètres. Les entrées se réduisent à :

- le calendrier : date, heure, jour de la semaine, fériés et ponts, vacances par zone avec leur
  début et leur fin (le premier samedi d'août n'est pas un samedi ordinaire) ;
- la couleur Bison Futé **par sens** (départs, retours) ;
- le scénario : pénétration VE, PDC par station (section 7) ;
- à court terme seulement : occupation récente et HERE temps réel.

**Rôle de Traffic Patterns.** Dans sa version standard, Traffic Patterns donne une vitesse typique par
segment, par jour de la semaine et par quart d'heure **(à vérifier dans le contrat)**. Il ne connaît
ni les vacances ni les chassés-croisés : c'est une ligne de base « semaine ordinaire ». L'effet des
vacances vient du calendrier et de Bison Futé, à travers ce que l'étape 1 a appris sur l'historique
HERE pluriannuel. La prévision dépend donc surtout de cet historique, pas de Traffic Patterns.

**Au-delà de 7 jours.** Rien ne bloque : le calendrier Bison Futé de l'année est publié à l'avance
(généralement en décembre) et les vacances scolaires sont connues longtemps avant. Le prochain
chassé-croisé est prévisible des mois à l'avance : calendrier → type de jour → trafic appris par axe →
occupation. On perd seulement l'information récente (météo, incidents, tendance des derniers jours),
d'où une restitution en médiane et 90ᵉ centile plutôt qu'en valeur ponctuelle. C'est l'usage visé
pour le dimensionnement et la préparation des week-ends rouges ou noirs.

### 8.2 Calendrier et géographie des vacances

- **Calendrier détaillé** : jour de la semaine, fériés, ponts, et vacances scolaires **par zone**
  (A, B, C), qui ne commencent pas le même jour.
- **Bison Futé** : la couleur est publiée par jour et par sens (départs, retours), au niveau national
  avec une distinction pour l'Île-de-France, et des précisions régionales surtout en hiver vers les
  Alpes **(à vérifier)**. Le temps réel est diffusé au format DATEX II ; l'existence d'une API pour
  le calendrier des couleurs reste à vérifier.
- **Géographie** : Bison Futé étant peu différencié par axe, le modèle doit apprendre où vont les
  vacanciers selon la période (février : A6, A7, A43 vers les Alpes ; été : A7, A9, A10, A63 vers le
  sud). On introduit des variables croisées axe × type de vacances × zone, en plus de l'identifiant
  de station. Chaque type de vacances doit apparaître au moins une fois dans l'historique : l'historique
  HERE pluriannuel compense pour la partie trafic.

### 8.3 Version macro : journées types

Une version agrégée sert de socle, de référence à battre et d'outil de planification :

1. **Typologie des jours** : semaine ordinaire, vendredi de départ, samedi rouge ou noir, dimanche de
   retour, férié, vacances ou non (par zone). Définie d'abord par le calendrier et Bison Futé, puis
   affinée par regroupement (k-moyennes) des profils journaliers de trafic HERE.
2. **Profils par station et par type de jour** : pour chaque heure $h$, la médiane et le
   90ᵉ centile de l'occupation,
   $$
   \hat y_{s,\theta}(h) = \operatorname{médiane}\left\{ y(s, t) : \theta(t) = \theta,\ h(t) = h \right\},
   \qquad
   \hat y^{90}_{s,\theta}(h) = Q_{0,9}\left\{ \cdots \right\}
   $$
   qui donnent un risque de saturation heure par heure.
3. **Scénarios** : la loi d'Erlang de la section 7 s'applique aux profils types, ce qui donne des
   cartes 2030 par type de jour.
4. **Horizon long** : c'est la version utilisée au-delà de J+7. Le type de jour se déduit du seul
   calendrier (vacances par zone, Bison Futé), donc à n'importe quel horizon.

Il faut quelques semaines par type de jour. Le profil moyen station × jour × heure de la section 5
en est la version la plus simple.

## 9. Données à réunir pour l'apprentissage réel

L'occupation n'existe que depuis mars 2026 environ : de mars à octobre, on couvre Pâques, les ponts
de mai, l'été et le début de la Toussaint, mais **pas les vacances d'hiver**. Les profils d'hiver
reposeront sur le trafic HERE des années précédentes et sur des hypothèses, à signaler comme tels.

| Priorité | Donnée | Profondeur | Remarque |
|---|---|---|---|
| 1 | Export Qualicharge complet (`charging_station_availability_snapshots`), même format | depuis le début de la collecte (mars 2026) jusqu'à aujourd'hui | au moins les 170 stations d'autoroute ; signaler les arrêts de collecte |
| 1 | Référentiel des stations à jour (`charging_stations.csv`) | dernière version | pour les stations ouvertes depuis juillet |
| 1 | HERE : vitesses historiques par segment, 15 min, avec vitesse fluide et nombre de sondes ou confiance | mars 2026 à aujourd'hui pour l'apprentissage ; 2024-2025 si possible pour l'hiver et les types de jours | classe fonctionnelle 1, 20 axes, deux sens (`note_here.md`, section 2) |
| 1 | HERE : référentiel des segments (TMC ou links, géométrie) | | pour la projection sur les axes |
| 2 | HERE : Traffic Patterns (vitesses typiques) | dernière version | entrée de la prévision à J+1 |
| 2 | HERE : clé de l'API Traffic Flow | | collecte temps réel, 15 min |
| 2 | Calendrier Bison Futé (couleurs par jour et par sens) | 2024 à 2026 | même profondeur que l'historique HERE |
| 2 | Vacances scolaires par zone et jours fériés | 2024 à 2026 | data.gouv.fr |
| 3 | Débits Avatar | mars à octobre 2026 (aujourd'hui mai à juillet) | compteurs interurbains seulement |
| 3 | Tracé exact des autoroutes (OSM, Route 500) | | améliore la détection des stations et les pk |

## 10. Limites et suite

- **Historique** : il faut l'export long de la collecte (section 9) ; sans vacances d'hiver dans
  l'occupation, les prévisions d'hiver resteront à prendre avec prudence.
- **Débits Avatar** : faible information hors cycle journalier ; à confirmer sur l'historique long,
  avec les seuls compteurs interurbains.
- **Tracé des autoroutes** : le tracé par villes étapes donne des pk à quelques km près et laisse
  échapper les stations nommées sans référence à l'aire. Un tracé OSM ou Route 500 améliorera
  la détection et les pk, et permettra un vrai graphe maillé (`aretes.csv`).
- **Biais d'exposition de l'étape 2** : elle est apprise sur le trafic observé mais utilisée avec le
  trafic prévu. Si l'écart pèse, il faudra l'apprendre sur des prévisions hors échantillon de
  l'étape 1.
- **Validation** : avec une seule saison estivale, une validation croisée par blocs de semaines sera
  plus fiable qu'un découpage unique.
- **Restitution** : l'interface Flask lira les prévisions (station, heure, taux prévu, couleur,
  probabilité de station pleine) et les profils de journées types.

## Annexe : commandes

```bash
pip install -r requirements.txt
# données synthétiques et apprentissage
python generer_donnees_synthetiques.py --sortie donnees_synthetiques
python entrainer.py --donnees donnees_synthetiques --horizon-h 24 --sortie resultats
# données réelles : placement des stations, puis conversion de l'historique
python reperer_stations_autoroute.py --referentiel charging_stations.csv --sortie donnees_reelles/stations.csv
python -c "import pandas as pd; from donnees import convertir_snapshots_qualicharge as c; \
c(pd.read_csv('charging_station_availability_snapshots.csv')).to_csv('occupation_bornes.csv', index=False)"
# débits Avatar : rattachement au graphe et corrélations
python rattacher_compteurs.py --donnees donnees_reelles --interurbains-seulement
# scénario de pénétration VE à 10 %
python scenarios.py --occupation donnees_reelles/occupation_bornes.csv --penetration 0.10 --sortie scenario_10pc.csv
```
