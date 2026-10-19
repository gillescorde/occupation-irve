"""
Restitution web du « Bison Futé des bornes » : carte de France (axes + stations), courbe
d'occupation horaire par station, calendrier avec week-ends / vacances / couleurs Bison Futé.

Usage :
    .venv\\Scripts\\python.exe webapp\\app.py
    puis ouvrir http://127.0.0.1:5000
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request

from calcul import (
    PENETRATION_REF, bornes_jour_complet, calculer_jour_complet, calculer_scenario,
    dates_calculables, obtenir_calcule_complet, preparer_contexte,
)

app = Flask(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
STATIONS_FILE = ROOT / "donnees_reelles" / "stations.csv"
AXES_FILE = ROOT / "donnees_reelles" / "axes_approx.csv"
OCCUPATION_FILE = ROOT / "donnees_reelles" / "occupation_bornes.csv"
CALENDRIER_FILE = ROOT / "donnees_reelles" / "calendrier" / "calendrier.csv"
REGIONS_FILE = Path(__file__).resolve().parent / "regions_france.geojson"

# Approximation des zones Bison Futé par région administrative (pas de table officielle
# département->zone trouvée ; zones réelles = 1 Île-de-France, 2 Grand-Ouest et Nord,
# 3 Bourgogne et Est, 4 Auvergne-Rhône-Alpes, 5 Sud-Ouest, 6 Arc méditerranéen, cf. bison_fute_pdf.py).
# Occitanie entière -> zone 6, Centre-Val de Loire -> zone 2 (choix fait avec l'utilisateur).
REGION_VERS_ZONE_BF = {
    "11": 1,  # Île-de-France
    "24": 2,  # Centre-Val de Loire
    "27": 3,  # Bourgogne-Franche-Comté
    "28": 2,  # Normandie
    "32": 2,  # Hauts-de-France
    "44": 3,  # Grand Est
    "52": 2,  # Pays de la Loire
    "53": 2,  # Bretagne
    "75": 5,  # Nouvelle-Aquitaine
    "76": 6,  # Occitanie
    "84": 4,  # Auvergne-Rhône-Alpes
    "93": 6,  # Provence-Alpes-Côte d'Azur
    "94": 6,  # Corse
}
NOM_ZONE_BF = {
    1: "Île-de-France", 2: "Grand-Ouest et Nord", 3: "Bourgogne et Est",
    4: "Auvergne-Rhône-Alpes", 5: "Sud-Ouest", 6: "Arc méditerranéen",
}

# ============================================================
# HELPERS
# ============================================================


def _clean(v):
    """NaN/NaT -> None (JSON null) ; scalaires numpy -> types Python natifs."""
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


# ============================================================
# CHARGEMENT DES DONNÉES (une fois, au démarrage)
# ============================================================


def charger_stations():
    return pd.read_csv(STATIONS_FILE)


def charger_axes():
    return pd.read_csv(AXES_FILE).sort_values(["axe", "pk"])


def charger_occupation():
    """Pivot (station_id, date) -> taux par heure 0..23, formule exacte de donnees.py:220-224."""
    occ = pd.read_csv(OCCUPATION_FILE, parse_dates=["horodatage"])
    occ["taux"] = (
        occ["nb_pdc_occupes"] / occ["nb_pdc_statut_connu"].where(occ["nb_pdc_statut_connu"] > 0)
    ).clip(0, 1)
    occ["date"] = occ["horodatage"].dt.strftime("%Y-%m-%d")
    occ["heure"] = occ["horodatage"].dt.hour
    pivot = occ.pivot_table(index=["station_id", "date"], columns="heure", values="taux")
    pivot = pivot.reindex(columns=range(24))
    return pivot, set(occ["date"].unique())


def charger_calendrier():
    return pd.read_csv(CALENDRIER_FILE).set_index("date")


def charger_zones():
    """Régions administratives (gregoiredavid/france-geojson) taguées par zone Bison Futé
    approchée. Une région par Feature (pas de fusion), coloriées par zone côté client."""
    import json

    geo = json.loads(REGIONS_FILE.read_text(encoding="utf-8"))
    for f in geo["features"]:
        code = f["properties"].get("code")
        zone = REGION_VERS_ZONE_BF.get(code)
        f["properties"] = {
            "code_region": code,
            "nom_region": f["properties"].get("nom"),
            "zone_bf": zone,
            "nom_zone_bf": NOM_ZONE_BF.get(zone),
        }
    return geo


STATIONS = charger_stations()
AXES = charger_axes()
OCCUPATION, DATES_DISPONIBLES = charger_occupation()
CALENDRIER = charger_calendrier()
ZONES_GEOJSON = charger_zones()

print("Préparation du contexte GNN (chargement + inférence préparatoire)...", flush=True)
CONTEXTE = preparer_contexte()  # gardé en mémoire : réutilisé par /api/scenario (volet prédiction)
CALCULE = obtenir_calcule_complet(CONTEXTE)
DATES_CALCULABLES = dates_calculables(CONTEXTE)
PREMIER_JOUR_COMPLET, DERNIER_JOUR_COMPLET = bornes_jour_complet(CONTEXTE)

STATIONS_AVEC_DONNEES = set(OCCUPATION.index.get_level_values(0))
NB_PDC = STATIONS.set_index("station_id")["nb_pdc"]

print(
    f"Chargé : {len(STATIONS)} stations, {AXES['axe'].nunique()} axes, "
    f"{len(DATES_DISPONIBLES)} jours de mesure, "
    f"{len(set(CALCULE.index.get_level_values(0)))} stations avec calcul GNN, "
    f"{len(DATES_CALCULABLES)} jours utilisables pour le volet prédiction",
    flush=True,
)

# ============================================================
# ROUTES
# ============================================================


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/prediction")
def prediction():
    return render_template("prediction.html")


@app.route("/api/stations")
def api_stations():
    features = []
    for _, r in STATIONS.iterrows():
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
                "properties": {
                    "station_id": r["station_id"],
                    "nom": r["nom"],
                    "axe": r["axe"],
                    "sens": _clean(r["sens"]),
                    "pk": _clean(r["pk"]),
                    "nb_pdc": _clean(r["nb_pdc"]),
                    "a_des_donnees": r["station_id"] in STATIONS_AVEC_DONNEES,
                },
            }
        )
    return jsonify({"type": "FeatureCollection", "features": features})


@app.route("/api/axes")
def api_axes():
    features = []
    for axe, g in AXES.groupby("axe", sort=False):
        coords = g[["lon", "lat"]].values.tolist()
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {"axe": axe},
            }
        )
    return jsonify({"type": "FeatureCollection", "features": features})


@app.route("/api/zones")
def api_zones():
    return jsonify(ZONES_GEOJSON)


@app.route("/api/config")
def api_config():
    return jsonify({
        "penetration_ref": PENETRATION_REF,
        "premier_jour_complet": PREMIER_JOUR_COMPLET.strftime("%Y-%m-%d"),
        "dernier_jour_complet": DERNIER_JOUR_COMPLET.strftime("%Y-%m-%d"),
    })


@app.route("/api/calendrier")
def api_calendrier():
    out = []
    for date, r in CALENDRIER.iterrows():
        jour = {
            "date": date,
            "jour_semaine": int(r["jour_semaine"]),
            "vac_a": bool(r["vac_a"]),
            "vac_b": bool(r["vac_b"]),
            "vac_c": bool(r["vac_c"]),
            "nom_vacances": r["nom_vacances"] if isinstance(r["nom_vacances"], str) and r["nom_vacances"] else None,
            "ferie": bool(r["ferie"]),
            "pont": bool(r["pont"]),
            "bf_depart": _clean(r["bf_depart"]),
            "bf_retour": _clean(r["bf_retour"]),
            "a_des_donnees": date in DATES_DISPONIBLES,
            "a_du_calcul": date in DATES_CALCULABLES,
        }
        for z in range(1, 7):
            jour[f"bf_depart_z{z}"] = _clean(r[f"bf_depart_z{z}"])
            jour[f"bf_retour_z{z}"] = _clean(r[f"bf_retour_z{z}"])
        out.append(jour)
    return jsonify(out)


@app.route("/api/occupation/<station_id>")
def api_occupation(station_id):
    date = request.args.get("date", "")
    if len(date) != 10:
        abort(400, "paramètre ?date=YYYY-MM-DD requis")
    if station_id not in STATIONS["station_id"].values:
        abort(404, "station inconnue")

    nb_pdc = int(NB_PDC.get(station_id, 0))

    def en_pdc(pivot):
        try:
            taux = pivot.loc[(station_id, date)]
        except KeyError:
            return [None] * 24
        return [None if pd.isna(v) else round(float(v) * nb_pdc, 1) for v in taux.tolist()]

    mesure_pdc = en_pdc(OCCUPATION)
    calcule_pdc = en_pdc(CALCULE)

    return jsonify({
        "heures": list(range(24)),
        "nb_pdc": nb_pdc,
        "mesure_pdc": mesure_pdc,
        "calcule_pdc": calcule_pdc,
    })


@app.route("/api/occupation_jour")
def api_occupation_jour():
    """Taux d'occupation moyen (mesuré) de chaque station pour une date donnée, pour colorer la
    carte d'un coup d'œil."""
    date = request.args.get("date", "")
    if len(date) != 10:
        abort(400, "paramètre ?date=YYYY-MM-DD requis")
    try:
        jour = OCCUPATION.xs(date, level="date")
    except KeyError:
        return jsonify({})
    moyenne = jour.mean(axis=1, skipna=True)
    return jsonify({sid: _clean(v) for sid, v in moyenne.items() if not pd.isna(v)})


@app.route("/api/scenario", methods=["POST"])
def api_scenario():
    """Volet prédiction : rejoue le GNN sur un jour historique avec des couleurs Bison Futé
    hypothétiques, et une pénétration VE optionnelle (sinon celle de l'historique, ~3 %).
    Body JSON : {"date": "YYYY-MM-DD", "niveaux": {"1": 0..3, ..., "6": 0..3}, "penetration": 0.10}."""
    body = request.get_json(silent=True) or {}
    date = body.get("date", "")
    niveaux = body.get("niveaux", {})
    if len(date) != 10 or set(map(str, range(1, 7))) != set(niveaux.keys()):
        abort(400, "attendu : {date: YYYY-MM-DD, niveaux: {1..6: 0..3}}")
    try:
        niveaux_zone = {int(z): int(n) for z, n in niveaux.items()}
    except (TypeError, ValueError):
        abort(400, "niveaux doit être un entier 0..3 par zone")
    if date not in DATES_CALCULABLES:
        abort(400, "pas assez d'historique réel pour cette date")

    penetration = body.get("penetration")
    if penetration is not None:
        try:
            penetration = float(penetration)
        except (TypeError, ValueError):
            abort(400, "penetration doit être un nombre (ex. 0.10 pour 10 %)")
        if not 0 < penetration <= 1:
            abort(400, "penetration doit être entre 0 et 1")

    resultat = calculer_scenario(CONTEXTE, date, niveaux_zone, penetration)
    return jsonify({"heures": list(range(24)), "par_station": resultat})


@app.route("/api/jour_complet", methods=["POST"])
def api_jour_complet():
    """Volet prédiction complète : jour au-delà de l'historique réel (ex. jour férié à venir),
    sans trafic ni occupation mesurés ce jour-là — prévus en cascade depuis la fin des données
    réelles. Scénario Bison Futé et pénétration VE optionnels (sinon Bison Futé officiel publié).
    Body JSON : {"date": "YYYY-MM-DD", "niveaux": {"1": 0..3, ..., "6": 0..3} (optionnel),
    "penetration": 0.10 (optionnel)}."""
    body = request.get_json(silent=True) or {}
    date = body.get("date", "")
    if len(date) != 10:
        abort(400, "paramètre date (YYYY-MM-DD) requis")

    niveaux = body.get("niveaux")
    niveaux_zone = None
    if niveaux is not None:
        if set(map(str, range(1, 7))) != set(niveaux.keys()):
            abort(400, "niveaux doit couvrir les zones 1..6")
        try:
            niveaux_zone = {int(z): int(n) for z, n in niveaux.items()}
        except (TypeError, ValueError):
            abort(400, "niveaux doit être un entier 0..3 par zone")

    penetration = body.get("penetration")
    if penetration is not None:
        try:
            penetration = float(penetration)
        except (TypeError, ValueError):
            abort(400, "penetration doit être un nombre (ex. 0.10 pour 10 %)")
        if not 0 < penetration <= 1:
            abort(400, "penetration doit être entre 0 et 1")

    resultat = calculer_jour_complet(CONTEXTE, date, niveaux_zone, penetration)
    if resultat is None:
        abort(400, "date hors de la plage prédictible (déjà couverte par l'historique réel, ou "
                    "au-delà du calendrier Bison Futé connu)")
    return jsonify(resultat)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
