"""
Tracés des axes à partir d'OpenStreetMap, en remplacement des villes étapes de
`reperer_stations_autoroute.py` (précision : quelques centaines de mètres au lieu de quelques km).

Entrées (réponses Overpass brutes, « out geom ») dans osm/brut/ :
  - axe_<AXE>.json   : voies highway=motorway|trunk portant la référence de l'axe ;
  - echangeurs.json  : nœuds highway=motorway_junction ;
  - aires.json       : nœuds ou voies highway=services|rest_area (facultatif).

Méthode, sans reconstruire la topologie OSM (bretelles, chaussées séparées, tronçons communs) :
  1. chaque point des voies OSM est projeté sur le tracé approché de l'axe (villes étapes) pour
     obtenir un pk provisoire ; les points à plus de `--ecart-max-km` sont écartés (homonymes,
     antennes) ;
  2. on regroupe par tranche de 1 km de pk et on prend la médiane des coordonnées : les deux
     chaussées se moyennent en un axe central ;
  3. le pk final est la longueur cumulée de cette ligne lissée, donc une vraie distance routière.
Les échangeurs et les aires à moins de 1 km du nouveau tracé reçoivent un pk.

Sorties : axes_osm.csv (axe, pk, lat, lon), meme format qu'axes_approx.csv,
          echangeurs.csv (axe, pk, nom, ref, lat, lon), aires_osm.csv (idem).
Usage :
    python traces_osm.py --brut donnees_reelles/osm/brut --sortie donnees_reelles
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reperer_stations_autoroute import AXES, haversine, projeter_sur_axe, trace


def points_voies(chemin):
    d = json.load(open(chemin, encoding="utf-8"))
    pts = [(p["lat"], p["lon"]) for e in d.get("elements", []) if e.get("type") == "way"
           for p in e.get("geometry", [])]
    return np.array(pts) if pts else np.zeros((0, 2))


def points_noeuds(chemin):
    if not Path(chemin).exists():
        return pd.DataFrame(columns=["lat", "lon", "nom", "ref", "type"])
    d = json.load(open(chemin, encoding="utf-8"))
    lignes = []
    for e in d.get("elements", []):
        t = e.get("tags", {})
        if e.get("type") == "node":
            lat, lon = e["lat"], e["lon"]
        elif "center" in e:
            lat, lon = e["center"]["lat"], e["center"]["lon"]
        elif e.get("geometry"):
            g = e["geometry"]
            lat, lon = np.mean([p["lat"] for p in g]), np.mean([p["lon"] for p in g])
        else:
            continue
        lignes.append(dict(lat=lat, lon=lon, nom=t.get("name", ""), ref=t.get("ref", ""),
                           type=t.get("highway", "")))
    return pd.DataFrame(lignes)


def projeter_lot(lat, lon, tr):
    """Projection vectorisée de nombreux points sur un tracé (distance km, pk)."""
    out = np.array([projeter_sur_axe(a, o, tr)[:2] for a, o in zip(lat, lon)])
    return out[:, 0], out[:, 1]


def trace_osm(axe, pts, ecart_max_km=3.0, pas_km=1.0):
    approx = trace(axe)[0]
    pts = pts[:: max(1, len(pts) // 20000)]  # sous-échantillonnage : la médiane par km suffit
    d, pk = projeter_lot(pts[:, 0], pts[:, 1], approx)
    ok = d <= ecart_max_km
    if ok.sum() < 10:
        return None, 0.0
    df = pd.DataFrame({"tranche": (pk[ok] // pas_km).astype(int), "lat": pts[ok, 0], "lon": pts[ok, 1]})
    g = df.groupby("tranche")[["lat", "lon"]].median().sort_index()
    couverture = len(g) * pas_km / max(approx["pk"].max(), 1)
    lat, lon = g["lat"].values, g["lon"].values
    pk_reel = np.concatenate([[0], np.cumsum(haversine(lat[:-1], lon[:-1], lat[1:], lon[1:]))])
    return pd.DataFrame({"axe": axe, "pk": pk_reel.round(3), "lat": lat.round(6), "lon": lon.round(6)}), couverture


def placer(noeuds, traces, distance_max_km=1.0):
    lignes = []
    for r in noeuds.itertuples():
        best = None
        for axe, tr in traces.items():
            d, pk, _ = projeter_sur_axe(r.lat, r.lon, tr)
            if d <= distance_max_km and (best is None or d < best[0]):
                best = (d, axe, pk)
        if best:
            lignes.append(dict(axe=best[1], pk=round(best[2], 2), nom=r.nom, ref=r.ref, type=r.type,
                               lat=r.lat, lon=r.lon, distance_axe_km=round(best[0], 2)))
    return pd.DataFrame(lignes).sort_values(["axe", "pk"]) if lignes else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brut", default="donnees_reelles/osm/brut")
    ap.add_argument("--sortie", default="donnees_reelles")
    ap.add_argument("--ecart-max-km", type=float, default=3.0)
    a = ap.parse_args()
    brut, out = Path(a.brut), Path(a.sortie)
    traces, manquants = {}, []
    for axe in AXES:
        f = brut / f"axe_{axe}.json"
        if not f.exists():
            manquants.append(axe)
            continue
        tr, couv = trace_osm(axe, points_voies(f), a.ecart_max_km)
        if tr is None:
            manquants.append(axe)
            continue
        traces[axe] = tr
        print(f"{axe:5s} {tr['pk'].max():6.0f} km, couverture du tracé approché {couv:.0%}")
    if manquants:
        print("sans tracé OSM (on garde le tracé approché) :", ", ".join(manquants))
        for axe in manquants:
            traces[axe] = trace(axe)[0]
    pd.concat(traces.values()).to_csv(out / "axes_osm.csv", index=False)
    for nom, fichier in (("echangeurs", "echangeurs.json"), ("aires_osm", "aires.json")):
        p = placer(points_noeuds(brut / fichier), traces)
        p.to_csv(out / f"{nom}.csv", index=False)
        print(f"{nom} : {len(p)} placés sur les axes")


if __name__ == "__main__":
    main()
