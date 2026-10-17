"""
Repère les stations d'autoroute dans le référentiel des stations (export `charging_stations.csv` de
la collecte Qualicharge) et écrit `stations.csv` au format attendu : station_id, nom, lat, lon,
axe, sens, pk, nb_pdc, puissance_kw.

Faute de tracé exact des autoroutes, chaque axe est décrit par une suite de villes étapes
(coordonnées approchées, à quelques km près). Les pk obtenus sont donc approximatifs (erreur de
quelques km), ce qui suffit pour un graphe où les aires sont espacées de 20 à 60 km. Dès qu'un vrai
tracé (OSM, Route 500) est disponible, `preparer_stations.py` le remplace.

Méthode :
  1. Axe : lu dans le nom quand il y figure (« Electra A10 - Aire de Cézac », « Aire de Boutroux_A10 »,
     « RN165 ») ; sinon, pour un nom d'aire (« Aire de ... »), l'axe approché le plus proche à moins de
     5 km ; sinon, une station proche d'une sortie (au moins 150 kW et 6 PDC, à moins de 2 km du tracé
     approché, hors grandes agglomérations : Tesla, Ionity, Fastned... en zone d'activité). C'est aussi la seule recharge des
     nationales, sans aires de service. Ces stations ont sens = 0 (elles servent les deux sens).
  2. pk : projection sur la ligne brisée des villes étapes de cet axe.
  3. Sens (1 = pk croissants, cf. ordre des villes ci-dessous) :
       - « direction <ville> » dans le nom : comparé au pk de la ville ;
       - suffixe Nord / Sud / Est / Ouest : côté de la chaussée (on roule à droite) ;
       - sinon 0 = inconnu ou station desservant les deux sens.
  Chaque station porte `methode_axe` et `methode_sens` pour savoir ce qui est à vérifier.

Usage :
    python reperer_stations_autoroute.py --referentiel charging_stations.csv --sortie donnees/stations.csv
"""
import argparse
import re
import unicodedata

import numpy as np
import pandas as pd

# Villes étapes (lat, lon), dans le sens des pk croissants. Coordonnées approchées.
AXES = {
    "A1": [("Paris", 48.90, 2.36), ("Roissy", 49.00, 2.53), ("Senlis", 49.21, 2.58), ("Ressons", 49.54, 2.70),
           ("Roye", 49.70, 2.79), ("Peronne", 49.90, 2.83), ("Arras", 50.29, 2.78), ("Lille", 50.63, 3.06)],
    "A4": [("Paris", 48.83, 2.39), ("Marne-la-Vallee", 48.85, 2.70), ("Meaux", 48.95, 2.88),
           ("Chateau-Thierry", 49.04, 3.40), ("Reims", 49.24, 4.03), ("Sainte-Menehould", 49.09, 4.90),
           ("Verdun", 49.16, 5.38), ("Metz", 49.12, 6.17), ("Saint-Avold", 49.10, 6.70),
           ("Phalsbourg", 48.77, 7.26), ("Strasbourg", 48.58, 7.75)],
    "A5": [("Senart", 48.58, 2.62), ("Montereau", 48.38, 3.00), ("Sens", 48.20, 3.28), ("Troyes", 48.30, 4.08),
           ("Chaumont", 48.11, 5.14), ("Langres", 47.86, 5.33)],
    "A6": [("Paris", 48.82, 2.36), ("Evry", 48.63, 2.44), ("Fontainebleau", 48.34, 2.60), ("Nemours", 48.27, 2.70),
           ("Auxerre", 47.80, 3.57), ("Avallon", 47.49, 3.91), ("Pouilly-en-Auxois", 47.26, 4.55),
           ("Beaune", 47.02, 4.84), ("Chalon-sur-Saone", 46.78, 4.85), ("Tournus", 46.56, 4.91),
           ("Macon", 46.31, 4.83), ("Villefranche-sur-Saone", 45.99, 4.72), ("Lyon", 45.76, 4.84)],
    "A7": [("Lyon", 45.74, 4.82), ("Vienne", 45.52, 4.87), ("Saint-Rambert-d'Albon", 45.30, 4.81),
           ("Valence", 44.93, 4.89), ("Montelimar", 44.56, 4.75), ("Bollene", 44.28, 4.75), ("Orange", 44.14, 4.81),
           ("Avignon", 43.99, 4.88), ("Cavaillon", 43.84, 5.04), ("Salon-de-Provence", 43.64, 5.10),
           ("Marseille", 43.35, 5.36)],
    "A8": [("Aix-en-Provence", 43.53, 5.45), ("Saint-Maximin", 43.45, 5.86), ("Brignoles", 43.40, 6.06),
           ("Le Luc", 43.39, 6.31), ("Frejus", 43.45, 6.74), ("Mandelieu", 43.55, 6.94), ("Nice", 43.70, 7.20),
           ("Menton", 43.78, 7.50)],
    "A9": [("Orange", 44.14, 4.81), ("Remoulins", 43.94, 4.56), ("Nimes", 43.84, 4.36), ("Montpellier", 43.60, 3.88),
           ("Gigean", 43.50, 3.71), ("Beziers", 43.34, 3.22), ("Narbonne", 43.18, 3.00), ("Perpignan", 42.70, 2.89),
           ("Le Boulou", 42.52, 2.83), ("Le Perthus", 42.46, 2.86)],
    "A10": [("Paris", 48.82, 2.32), ("Massy", 48.73, 2.27), ("Ablis", 48.52, 1.85), ("Artenay", 48.08, 1.88),
            ("Orleans", 47.90, 1.90), ("Blois", 47.59, 1.33), ("Tours", 47.39, 0.69), ("Chatellerault", 46.82, 0.55),
            ("Poitiers", 46.58, 0.34), ("Niort", 46.32, -0.46), ("Saintes", 45.75, -0.63), ("Bordeaux", 44.90, -0.53)],
    "A11": [("Ponthevrard", 48.55, 1.92), ("Chartres", 48.44, 1.49), ("Nogent-le-Rotrou", 48.32, 0.82),
            ("Le Mans", 48.00, 0.20), ("Sable", 47.84, -0.33), ("Angers", 47.47, -0.55), ("Ancenis", 47.37, -1.18),
            ("Nantes", 47.25, -1.50)],
    "A13": [("Paris", 48.85, 2.26), ("Orgeval", 48.92, 1.98), ("Mantes", 48.99, 1.71), ("Vernon", 49.07, 1.48),
            ("Rouen", 49.37, 1.07), ("Pont-l'Eveque", 49.28, 0.18), ("Caen", 49.18, -0.37)],
    "A20": [("Vierzon", 47.22, 2.07), ("Chateauroux", 46.81, 1.69), ("Argenton", 46.59, 1.52),
            ("La Souterraine", 46.24, 1.49), ("Limoges", 45.83, 1.26), ("Uzerche", 45.42, 1.56), ("Brive", 45.15, 1.53),
            ("Souillac", 44.90, 1.48), ("Cahors", 44.45, 1.44), ("Caussade", 44.16, 1.54), ("Montauban", 44.02, 1.35)],
    "A26": [("Calais", 50.93, 1.86), ("Saint-Omer", 50.75, 2.25), ("Bethune", 50.53, 2.64), ("Arras", 50.29, 2.78),
            ("Cambrai", 50.17, 3.23), ("Saint-Quentin", 49.85, 3.29), ("Laon", 49.56, 3.62), ("Reims", 49.24, 4.03),
            ("Chalons-en-Champagne", 48.96, 4.36), ("Troyes", 48.30, 4.08)],
    "A31": [("Luxembourg", 49.46, 6.13), ("Thionville", 49.36, 6.17), ("Metz", 49.12, 6.17), ("Nancy", 48.69, 6.18),
            ("Toul", 48.68, 5.89), ("Neufchateau", 48.36, 5.70), ("Langres", 47.86, 5.33), ("Dijon", 47.32, 5.04),
            ("Beaune", 47.02, 4.84)],
    "A36": [("Beaune", 47.02, 4.84), ("Seurre", 47.00, 5.15), ("Dole", 47.09, 5.49), ("Besancon", 47.24, 6.02),
            ("Baume-les-Dames", 47.35, 6.36), ("Montbeliard", 47.51, 6.80), ("Mulhouse", 47.75, 7.34)],
    "A61": [("Toulouse", 43.57, 1.48), ("Villefranche-de-Lauragais", 43.40, 1.72), ("Castelnaudary", 43.32, 1.95),
            ("Carcassonne", 43.21, 2.35), ("Lezignan", 43.20, 2.76), ("Narbonne", 43.18, 3.00)],
    "A62": [("Bordeaux", 44.80, -0.55), ("Langon", 44.55, -0.25), ("Marmande", 44.50, 0.17), ("Agen", 44.20, 0.62),
            ("Castelsarrasin", 44.04, 1.11), ("Montauban", 44.02, 1.35), ("Toulouse", 43.65, 1.43)],
    "A63": [("Bordeaux", 44.78, -0.60), ("Belin", 44.50, -0.79), ("Labouheyre", 44.21, -0.92), ("Castets", 43.88, -1.14),
            ("Bayonne", 43.49, -1.47), ("Biarritz", 43.46, -1.53), ("Hendaye", 43.36, -1.77)],
    "A71": [("Orleans", 47.85, 1.93), ("Salbris", 47.43, 2.05), ("Vierzon", 47.22, 2.07), ("Bourges", 47.08, 2.40),
            ("Saint-Amand-Montrond", 46.72, 2.50), ("Montlucon", 46.34, 2.60), ("Riom", 45.89, 3.11),
            ("Clermont-Ferrand", 45.78, 3.08)],
    "A75": [("Clermont-Ferrand", 45.78, 3.08), ("Issoire", 45.54, 3.25), ("Massiac", 45.25, 3.20),
            ("Saint-Flour", 45.03, 3.09), ("Marvejols", 44.55, 3.29), ("Millau", 44.10, 3.08), ("Lodeve", 43.73, 3.32),
            ("Pezenas", 43.46, 3.42), ("Beziers", 43.37, 3.27)],
    "A89": [("Libourne", 44.92, -0.24), ("Mussidan", 45.04, 0.36), ("Perigueux", 45.18, 0.72), ("Brive", 45.15, 1.53),
            ("Tulle", 45.27, 1.77), ("Ussel", 45.55, 2.31), ("Clermont-Ferrand", 45.80, 3.05)],
    # Bretagne : voies express gratuites (2x2 voies) et A84
    "A84": [("Rennes", 48.14, -1.63), ("Saint-Aubin-du-Cormier", 48.26, -1.40), ("Fougeres", 48.36, -1.24),
            ("Avranches", 48.66, -1.33), ("Villedieu-les-Poeles", 48.83, -1.22), ("Pont-Farcy", 48.98, -1.03),
            ("Villers-Bocage", 49.08, -0.65), ("Caen", 49.15, -0.42)],
    "N12": [("Rennes", 48.13, -1.75), ("Montauban-de-Bretagne", 48.20, -2.05), ("Lamballe", 48.47, -2.51),
            ("Saint-Brieuc", 48.50, -2.76), ("Guingamp", 48.56, -3.15), ("Morlaix", 48.57, -3.83),
            ("Landerneau", 48.45, -4.25), ("Brest", 48.42, -4.47)],
    "N24": [("Rennes", 48.10, -1.75), ("Plelan-le-Grand", 48.00, -2.10), ("Ploermel", 47.93, -2.40),
            ("Josselin", 47.95, -2.55), ("Locmine", 47.88, -2.83), ("Baud", 47.87, -3.02), ("Lorient", 47.77, -3.33)],
    "N137": [("Saint-Malo", 48.63, -2.00), ("Tinteniac", 48.33, -1.83), ("Rennes", 48.12, -1.69),
             ("Bain-de-Bretagne", 47.84, -1.68), ("Derval", 47.67, -1.67), ("Nozay", 47.56, -1.62),
             ("Nantes", 47.26, -1.57)],
    "N165": [("Nantes", 47.24, -1.64), ("Savenay", 47.36, -1.94), ("La Roche-Bernard", 47.52, -2.30),
             ("Vannes", 47.67, -2.76), ("Auray", 47.67, -2.98), ("Lorient", 47.77, -3.33), ("Quimperle", 47.87, -3.55),
             ("Quimper", 47.98, -4.10), ("Chateaulin", 48.20, -4.09), ("Brest", 48.42, -4.42)],
    # Alpes
    "A40": [("Macon", 46.30, 4.88), ("Bourg-en-Bresse", 46.20, 5.25), ("Nantua", 46.15, 5.60),
            ("Bellegarde", 46.10, 5.83), ("Saint-Julien-en-Genevois", 46.14, 6.08), ("Annemasse", 46.18, 6.25),
            ("Bonneville", 46.08, 6.40), ("Cluses", 46.06, 6.58), ("Sallanches", 45.94, 6.63), ("Le Fayet", 45.91, 6.70)],
    "A41": [("Grenoble", 45.20, 5.75), ("Crolles", 45.28, 5.88), ("Pontcharra", 45.43, 6.02), ("Chambery", 45.57, 5.92),
            ("Aix-les-Bains", 45.69, 5.91), ("Rumilly", 45.86, 5.94), ("Annecy", 45.90, 6.11),
            ("Cruseilles", 46.03, 6.10), ("Geneve", 46.18, 6.10)],
    "A43": [("Lyon", 45.70, 4.95), ("L'Isle-d'Abeau", 45.62, 5.23), ("La Tour-du-Pin", 45.57, 5.44),
            ("Chambery", 45.57, 5.92), ("Montmelian", 45.50, 6.05), ("Aiguebelle", 45.54, 6.31),
            ("Saint-Jean-de-Maurienne", 45.28, 6.35), ("Modane", 45.20, 6.67)],
    "A48": [("Coiranne", 45.55, 5.38), ("Voiron", 45.37, 5.58), ("Voreppe", 45.29, 5.64), ("Grenoble", 45.19, 5.70)],
    "A49": [("Voreppe", 45.29, 5.64), ("Tullins", 45.30, 5.48), ("Saint-Marcellin", 45.15, 5.32),
            ("Romans", 45.05, 5.05), ("Valence", 44.95, 4.92)],
    "A51": [("Aix-en-Provence", 43.50, 5.43), ("Pertuis", 43.68, 5.48), ("Manosque", 43.83, 5.79),
            ("Sisteron", 44.19, 5.94), ("La Saulce", 44.43, 6.00)],
    "N85": [("Vizille", 45.08, 5.77), ("La Mure", 44.90, 5.78), ("Corps", 44.82, 5.95), ("Gap", 44.56, 6.08),
            ("Sisteron", 44.19, 5.94), ("Digne", 44.09, 6.24), ("Castellane", 43.85, 6.51), ("Grasse", 43.66, 6.92)],
    # N90 précédée de l'A430 (Montmélian - Albertville) : l'accès aux stations de Tarentaise
    "N90": [("Montmelian", 45.50, 6.05), ("Albertville", 45.66, 6.38), ("Moutiers", 45.48, 6.53),
            ("Aime", 45.55, 6.65), ("Bourg-Saint-Maurice", 45.62, 6.77)],
}

R = 6371.0


def sans_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def trace(axe, pas_km=1.0):
    """Ligne brisée densifiée : DataFrame axe, pk, lat, lon."""
    v = AXES[axe]
    lat, lon = np.array([p[1] for p in v]), np.array([p[2] for p in v])
    d = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
    pk_v = np.concatenate([[0], np.cumsum(d)])
    pk = np.arange(0, pk_v[-1], pas_km)
    return pd.DataFrame({"axe": axe, "pk": np.append(pk, pk_v[-1]),
                         "lat": np.interp(np.append(pk, pk_v[-1]), pk_v, lat),
                         "lon": np.interp(np.append(pk, pk_v[-1]), pk_v, lon)}), dict(zip((p[0] for p in v), pk_v))


def haversine(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def projeter_sur_axe(lat, lon, tr):
    """Renvoie (distance km, pk, vecteur direction unitaire (est, nord)) du point le plus proche du tracé."""
    x = np.radians(tr["lon"].values) * np.cos(np.radians(lat)) * R
    y = np.radians(tr["lat"].values) * R
    px, py = np.radians(lon) * np.cos(np.radians(lat)) * R, np.radians(lat) * R
    A = np.column_stack([x[:-1], y[:-1]])
    AB = np.column_stack([np.diff(x), np.diff(y)])
    L2 = np.maximum((AB ** 2).sum(1), 1e-9)
    t = np.clip(((np.array([px, py]) - A) * AB).sum(1) / L2, 0, 1)
    proj = A + t[:, None] * AB
    dist = np.hypot(proj[:, 0] - px, proj[:, 1] - py)
    j = dist.argmin()
    pk = tr["pk"].values[j] + t[j] * (tr["pk"].values[j + 1] - tr["pk"].values[j])
    return dist[j], pk, AB[j] / np.sqrt(L2[j])


# Grandes agglomérations : le tracé approché les traverse, et une station rapide y sert d'abord la ville
AGGLOS = {"Paris": (48.86, 2.35), "Lyon": (45.76, 4.84), "Marseille": (43.30, 5.37), "Toulouse": (43.60, 1.44),
          "Nice": (43.70, 7.26), "Nantes": (47.22, -1.55), "Montpellier": (43.61, 3.88), "Strasbourg": (48.58, 7.75),
          "Bordeaux": (44.84, -0.58), "Lille": (50.63, 3.06), "Rennes": (48.11, -1.68), "Grenoble": (45.19, 5.72),
          "Toulon": (43.12, 5.93), "Rouen": (49.44, 1.10), "Tours": (47.39, 0.69), "Orleans": (47.90, 1.91),
          "Clermont-Ferrand": (45.78, 3.08), "Dijon": (47.32, 5.04), "Metz": (49.12, 6.18), "Nancy": (48.69, 6.18),
          "Brest": (48.39, -4.49), "Caen": (49.18, -0.37), "Angers": (47.47, -0.55), "Le Mans": (48.00, 0.20),
          "Reims": (49.26, 4.03), "Avignon": (43.95, 4.81), "Perpignan": (42.70, 2.90), "Limoges": (45.83, 1.26),
          "Poitiers": (46.58, 0.34), "Besancon": (47.24, 6.02), "Mulhouse": (47.75, 7.34), "Annecy": (45.90, 6.13),
          "Aix-en-Provence": (43.53, 5.45), "Nimes": (43.84, 4.36), "Saint-Etienne": (45.44, 4.39)}

CARDINAUX = {"nord": (0, 1), "sud": (0, -1), "est": (1, 0), "ouest": (-1, 0)}


TOUTES_VILLES = {sans_accents(v[0]) for etapes in AXES.values() for v in etapes}


def pk_ville(texte, villes):
    for ville, pk_v in villes.items():
        v = sans_accents(ville)
        if texte.startswith(v) or (texte and v.startswith(texte.split()[0])):
            return pk_v
    return None


def sens_depuis_nom(nom, pk, direction, villes):
    n = sans_accents(nom).replace("_", " ")
    m = re.search(r"direction\s+([a-z' -]+)", n)
    if m:
        pk_c = pk_ville(m.group(1).strip(), villes)
        if pk_c is not None:
            return (1 if pk_c > pk else 2), "direction"
    m = re.search(r"\bsens\s+([a-z'-]+)\s+([a-z'-]+)", n)  # « Sens Paris Bordeaux », « Sens Paris Province »
    if m:
        pk_o, pk_d = pk_ville(m.group(1), villes), pk_ville(m.group(2), villes)
        if pk_d is not None:
            return (1 if pk_d > pk else 2), "direction"
        if pk_o is not None:
            return (1 if pk_o < pk else 2), "direction"
    m = re.search(r"\b([a-z'-]+)?\s*\b(nord|sud|est|ouest)\b", n)
    # « Toulouse Sud », « Montpellier Sud » désignent une ville, pas le côté de la chaussée
    if m and (m.group(1) or "") not in TOUTES_VILLES and m.group(1) not in ("du", "de", "des", "la"):
        m = re.search(r"\b(nord|sud|est|ouest)\b", n)
        cx, cy = CARDINAUX[m.group(1)]
        droite = (direction[1], -direction[0])  # à droite du sens des pk croissants
        return (1 if cx * droite[0] + cy * droite[1] > 0 else 2), "cote_chaussee"
    return 0, "inconnu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--referentiel", required=True, help="charging_stations.csv (export Qualicharge)")
    ap.add_argument("--sortie", default="stations.csv")
    ap.add_argument("--sortie-axes", default=None, help="écrit aussi le tracé approché (axe, pk, lat, lon)")
    ap.add_argument("--tolerance-axe-nomme-km", type=float, default=15.0)
    ap.add_argument("--tolerance-aire-km", type=float, default=5.0)
    ap.add_argument("--traces", help="axes_osm.csv (traces_osm.py) à la place des villes étapes")
    ap.add_argument("--echangeurs", help="echangeurs.csv (traces_osm.py) : stations de sortie près d'un échangeur")
    ap.add_argument("--puissance-min-sortie-kw", type=float, default=150.0)
    ap.add_argument("--pdc-min-sortie", type=int, default=6)
    ap.add_argument("--tolerance-sortie-km", type=float, default=2.0,
                    help="distance max au tracé, ou à l'échangeur le plus proche avec --echangeurs")
    ap.add_argument("--rayon-agglo-km", type=float, default=12.0,
                    help="pas de station de sortie si proche du centre d'une grande agglomération (trafic urbain)")
    args = ap.parse_args()

    ref = pd.read_csv(args.referentiel, low_memory=False)
    st = pd.DataFrame({
        "station_id": ref["station_key"], "nom": ref["station.nom_station"].fillna(""),
        "lat": ref["station.coordinates.lat"], "lon": ref["station.coordinates.lng"],
        "nb_pdc": ref["station.pdc_count"], "puissance_kw": ref["raw.summary.max_power"],
        "operateur": ref["station.nom_operateur"]})
    traces = {a: trace(a) for a in AXES}
    if args.traces:  # tracé OSM (traces_osm.py) : pk routiers ; pk des villes étapes reprojetés dessus
        osm = pd.read_csv(args.traces)
        for a, tr in osm.groupby("axe"):
            if a in AXES:
                tr = tr.sort_values("pk").reset_index(drop=True)
                traces[a] = (tr, {v: projeter_sur_axe(la, lo, tr)[1] for v, la, lo in AXES[a]})
    sorties = pd.read_csv(args.echangeurs) if args.echangeurs else None
    if args.sortie_axes:
        pd.concat([t for t, _ in traces.values()]).round(5).to_csv(args.sortie_axes, index=False)

    lignes = []
    for r in st.itertuples():
        n = sans_accents(r.nom).replace("_", " ")
        if re.search(r"trucks only|camions", n):  # bornes réservées aux poids lourds : autre clientèle
            continue
        nomme = re.findall(r"(?<![a-z0-9])(a|n|rn) ?(\d{1,3})(?![0-9])", n)
        axes_nommes = [a for a in dict.fromkeys(f"{t[-1].upper()}{x}" for t, x in nomme) if a in AXES]
        candidats = []
        if axes_nommes:
            for a in axes_nommes:
                d, pk, u = projeter_sur_axe(r.lat, r.lon, traces[a][0])
                if d <= args.tolerance_axe_nomme_km:
                    candidats.append((d, a, pk, u, "nom"))
        elif re.search(r"\baire\b", n) and "covoiturage" not in n:
            for a, (tr, _) in traces.items():
                d, pk, u = projeter_sur_axe(r.lat, r.lon, tr)
                if d <= args.tolerance_aire_km:
                    candidats.append((d, a, pk, u, "aire_proche_axe"))
        hors_agglo = all(haversine(r.lat, r.lon, la, lo) > args.rayon_agglo_km * (2 if v == "Paris" else 1)
                         for v, (la, lo) in AGGLOS.items())
        if (not candidats and hors_agglo and r.puissance_kw >= args.puissance_min_sortie_kw
                and r.nb_pdc >= args.pdc_min_sortie):
            # stations hors aire, près d'une sortie (Tesla, Ionity, Fastned... en zone d'activité) : seule
            # recharge des nationales, sans aires de service, et souvent moins chère que les aires
            for a, (tr, _) in traces.items():
                d, pk, u = projeter_sur_axe(r.lat, r.lon, tr)
                if sorties is not None:  # échangeurs OSM : distance à la sortie la plus proche de l'axe
                    e = sorties[sorties["axe"] == a]
                    d_sortie = haversine(r.lat, r.lon, e["lat"].values, e["lon"].values).min() if len(e) else np.inf
                    if d_sortie <= args.tolerance_sortie_km:
                        candidats.append((d, a, pk, u, "sortie_proche_axe"))
                elif d <= args.tolerance_sortie_km:
                    candidats.append((d, a, pk, u, "sortie_proche_axe"))
        if not candidats:
            continue
        d, a, pk, u, meth = min(candidats, key=lambda c: c[0])
        sens, meth_sens = sens_depuis_nom(r.nom, pk, u, traces[a][1])
        if meth == "sortie_proche_axe" and meth_sens == "inconnu":
            meth_sens = "hors_aire"  # sens 0 : une station de sortie sert les deux sens
        lignes.append(dict(station_id=r.station_id, nom=r.nom, lat=r.lat, lon=r.lon, axe=a, sens=sens,
                           pk=round(pk, 1), nb_pdc=r.nb_pdc, puissance_kw=r.puissance_kw, operateur=r.operateur,
                           distance_axe_approche_km=round(d, 1), methode_axe=meth, methode_sens=meth_sens))
    out = pd.DataFrame(lignes).sort_values(["axe", "pk"])
    out.to_csv(args.sortie, index=False)
    print(f"{len(out)} stations d'autoroute repérées sur {len(st)} -> {args.sortie}")
    print(out.groupby("methode_axe").size().to_string())
    print(out.groupby("methode_sens").size().to_string())
    print(out.groupby("axe").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
