"""
Client de l'API HERE Traffic Flow (v7) pour le « Bison Futé des bornes ».

Interroge l'état du trafic le long des autoroutes repérées, ne garde que les segments autoroutiers
qui suivent l'axe, leur attribue (axe, sens, pk) comme aux stations, puis agrège par tronçon de
10 km. Chaque appel ajoute une photographie horodatée aux fichiers de sortie : lancé toutes les 5 à
15 min (tâche planifiée), il constitue l'historique temps réel.

Ce qu'il faut vérifier dans la documentation HERE de votre contrat avant la mise en service :
taille maximale d'une bbox, quotas d'appels, et unité des vitesses (m/s dans la v7 publique).

Usage :
    Authentification via Mobicloud (~/.mobicloud/login.json), comme here_functions.py.
    python here_flux.py --axes donnees_reelles/axes_approx.csv --axe A10 --axe A7 --sortie donnees_reelles/here
    # sans réseau, pour tester l'analyse sur une réponse enregistrée :
    python here_flux.py --axes donnees_reelles/axes_approx.csv --axe A10 --reponse-json reponse.json --sortie test_here
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from here_functions import get_token
from reperer_stations_autoroute import haversine, projeter_sur_axe, trace

URL = "https://mobicloud.ifpen.com/geoservices/api/here/data.traffic.hereapi.com/v7/flow"


def bboxes_axe(tr: pd.DataFrame, pas_km=60.0, marge_deg=0.05):
    """Découpe le tracé d'un axe en boîtes successives (ouest, sud, est, nord) qui le recouvrent."""
    boites = []
    for debut in np.arange(0, tr["pk"].max(), pas_km):
        m = tr[(tr["pk"] >= debut) & (tr["pk"] <= debut + pas_km)]
        if len(m):
            boites.append((m["lon"].min() - marge_deg, m["lat"].min() - marge_deg,
                           m["lon"].max() + marge_deg, m["lat"].max() + marge_deg))
    return boites


def appeler(bbox, token, essais=4):
    import requests
    params = {"in": "bbox:{:.5f},{:.5f},{:.5f},{:.5f}".format(*bbox), "locationReferencing": "shape"}
    headers = {"Authorization": f"Bearer {token}"}
    for i in range(essais):
        r = requests.get(URL, params=params, headers=headers, timeout=30)
        if r.status_code == 429:  # quota par seconde dépassé : on attend et on réessaie
            time.sleep(2 ** i)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def segments(reponse: dict) -> pd.DataFrame:
    """Une ligne par segment HERE : géométrie, vitesses (km/h), jam factor, confiance."""
    lignes = []
    for res in reponse.get("results", []):
        loc, flux = res.get("location", {}), res.get("currentFlow", {})
        pts = [p for lien in loc.get("shape", {}).get("links", []) for p in lien.get("points", [])]
        if len(pts) < 2 or "freeFlow" not in flux:
            continue
        lignes.append(dict(
            description=loc.get("description", ""), longueur_m=loc.get("length", np.nan),
            lat0=pts[0]["lat"], lon0=pts[0]["lng"], lat1=pts[-1]["lat"], lon1=pts[-1]["lng"],
            lat_mil=pts[len(pts) // 2]["lat"], lon_mil=pts[len(pts) // 2]["lng"],
            vitesse_kmh=3.6 * flux.get("speedUncapped", flux.get("speed", np.nan)),
            vitesse_fluide_kmh=3.6 * flux["freeFlow"], jam_factor=flux.get("jamFactor", np.nan),
            confiance=flux.get("confidence", np.nan), praticable=flux.get("traversability", "")))
    return pd.DataFrame(lignes)


def placer_sur_axe(seg: pd.DataFrame, axe: str, tr: pd.DataFrame, distance_max_km=3.0,
                   vitesse_fluide_min=90.0, cos_min=0.8) -> pd.DataFrame:
    """Garde les segments autoroutiers parallèles à l'axe : proches du tracé, vitesse fluide d'autoroute,
    orientés comme l'axe. Le sens vient de l'orientation du segment (1 = vers les pk croissants)."""
    lignes = []
    for r in seg.itertuples():
        if r.vitesse_fluide_kmh < vitesse_fluide_min:
            continue
        d, pk, u = projeter_sur_axe(r.lat_mil, r.lon_mil, tr)
        if d > distance_max_km:
            continue
        v = np.array([(r.lon1 - r.lon0) * np.cos(np.radians(r.lat_mil)), r.lat1 - r.lat0])
        n = np.linalg.norm(v)
        if n == 0:
            continue
        cos = float(v @ u / n)
        if abs(cos) < cos_min:
            continue
        lignes.append({**r._asdict(), "axe": axe, "sens": 1 if cos > 0 else 2, "pk": round(pk, 2),
                       "distance_axe_km": round(d, 2)})
    return pd.DataFrame(lignes).drop(columns="Index", errors="ignore")


def agreger_troncons(seg: pd.DataFrame, horodatage, longueur_troncon_km=10.0) -> pd.DataFrame:
    """Agrège par (axe, sens, tronçon). La vitesse du tronçon est la vitesse moyenne de parcours :
    longueur totale / temps total, ce qui pondère correctement un court segment très lent."""
    s = seg.dropna(subset=["vitesse_kmh"]).copy()
    s["troncon_pk"] = (s["pk"] // longueur_troncon_km) * longueur_troncon_km
    s["L"] = s["longueur_m"].fillna(100.0) / 1000
    s["t_h"] = s["L"] / s["vitesse_kmh"].clip(lower=1)
    s["t_fluide_h"] = s["L"] / s["vitesse_fluide_kmh"].clip(lower=1)
    s["jam_L"] = s["jam_factor"] * s["L"]
    s["conf_L"] = s["confiance"] * s["L"]
    g = s.groupby(["axe", "sens", "troncon_pk"]).agg(L=("L", "sum"), t=("t_h", "sum"), tf=("t_fluide_h", "sum"),
                                                     jam=("jam_L", "sum"), conf=("conf_L", "sum"),
                                                     nb_segments=("L", "size")).reset_index()
    return pd.DataFrame({
        "horodatage": horodatage, "axe": g["axe"], "sens": g["sens"], "troncon_pk": g["troncon_pk"],
        "longueur_couverte_km": g["L"].round(2), "nb_segments": g["nb_segments"],
        "vitesse_kmh": (g["L"] / g["t"]).round(1), "vitesse_fluide_kmh": (g["L"] / g["tf"]).round(1),
        "congestion": (1 - g["tf"] / g["t"]).clip(lower=0).round(3),  # 1 - v / v_fluide
        "temps_parcours_min": (60 * g["t"]).round(2), "jam_factor": (g["jam"] / g["L"]).round(2),
        "confiance": (g["conf"] / g["L"]).round(2)})


def ajouter_csv(df: pd.DataFrame, chemin: Path):
    df.to_csv(chemin, mode="a", header=not chemin.exists(), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", required=True, help="tracé (axe, pk, lat, lon), ex. donnees_reelles/axes_approx.csv")
    ap.add_argument("--axe", action="append", required=True, help="axe à interroger (répétable)")
    ap.add_argument("--sortie", default="here")
    ap.add_argument("--reponse-json", help="analyser une réponse enregistrée au lieu d'appeler l'API")
    ap.add_argument("--garder-brut", action="store_true", help="enregistrer aussi chaque segment")
    args = ap.parse_args()
    out = Path(args.sortie)
    out.mkdir(parents=True, exist_ok=True)
    axes = pd.read_csv(args.axes)
    token = None if args.reponse_json else get_token()

    horodatage = pd.Timestamp.now(tz="Europe/Paris").tz_localize(None).floor("min")
    for axe in args.axe:
        tr = axes[axes["axe"] == axe].sort_values("pk")
        if tr.empty:
            tr = trace(axe)[0]
        if args.reponse_json:
            reponses = [json.load(open(args.reponse_json, encoding="utf-8"))]
        else:
            reponses = [appeler(b, token) for b in bboxes_axe(tr)]
        seg = pd.concat([segments(r) for r in reponses], ignore_index=True)
        if seg.empty:
            print(f"{axe} : aucun segment renvoyé")
            continue
        seg = seg.drop_duplicates(subset=["lat0", "lon0", "lat1", "lon1"])  # les bbox se chevauchent
        sur_axe = placer_sur_axe(seg, axe, tr)
        if sur_axe.empty:
            print(f"{axe} : {len(seg)} segments reçus, aucun sur l'autoroute")
            continue
        agg = agreger_troncons(sur_axe, horodatage)
        ajouter_csv(agg, out / "here_troncons.csv")
        if args.garder_brut:
            ajouter_csv(sur_axe.assign(horodatage=horodatage), out / "here_segments.csv")
        print(f"{axe} : {len(seg)} segments reçus, {len(sur_axe)} sur l'autoroute, {len(agg)} tronçons "
              f"(congestion moyenne {agg['congestion'].mean():.2f})")


if __name__ == "__main__":
    main()
