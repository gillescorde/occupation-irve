"""
Construit `stations.csv` (station_id, lat, lon, axe, sens, pk, nb_pdc, puissance_kw) pour les stations
présentes dans l'historique d'occupation, en les plaçant sur les autoroutes.

Entrées :
  --irve-statique  fichier consolidé IRVE statique (data.gouv.fr, « Fichier consolidé des Bornes de
                   Recharge pour Véhicules Électriques »), une ligne par point de charge. Colonnes
                   utilisées : id_station_itinerance, id_pdc_itinerance, nom_station,
                   consolidated_latitude / consolidated_longitude (ou coordonneesXY), puissance_nominale.
  --axes           tracé des axes : axe, pk, lat, lon (une ligne par point, dans l'ordre des pk).
                   Un point tous les 1 à 5 km suffit (extrait d'OSM ou de Route 500).
  --occupation     occupation_bornes.csv, pour ne garder que les stations qui ont un historique.

Chaque station est projetée sur le segment d'axe le plus proche. Elle est retenue si elle est à moins
de --distance-max-m. Le côté de la chaussée donne le sens : on roule à droite, donc une aire à droite
du sens des pk croissants est en sens 1, à gauche en sens 2. Une station sur un échangeur ou à
cheval sur les deux sens tombera d'un côté arbitraire : à vérifier sur carte pour les grosses stations.

Usage :
    python preparer_stations.py --irve-statique irve_statique.csv --axes axes.csv \\
        --occupation donnees/occupation_bornes.csv --sortie donnees/stations.csv
"""
import argparse

import numpy as np
import pandas as pd

R_TERRE = 6_371_000.0


def stations_irve(statique: pd.DataFrame) -> pd.DataFrame:
    s = statique.copy()
    if "consolidated_latitude" not in s and "coordonneesXY" in s:
        xy = s["coordonneesXY"].astype(str).str.strip("[] ").str.split(",", expand=True).astype(float)
        s["consolidated_longitude"], s["consolidated_latitude"] = xy[0], xy[1]
    g = s.groupby("id_station_itinerance").agg(
        nom=("nom_station", "first"), lat=("consolidated_latitude", "median"),
        lon=("consolidated_longitude", "median"), nb_pdc=("id_pdc_itinerance", "nunique"),
        puissance_kw=("puissance_nominale", "max"))
    return g.reset_index().rename(columns={"id_station_itinerance": "station_id"})


def projeter(stations: pd.DataFrame, axes: pd.DataFrame, distance_max_m=300.0) -> pd.DataFrame:
    lat0 = np.radians(axes["lat"].mean())

    def xy(lat, lon):  # projection équirectangulaire locale, en mètres
        return np.column_stack([np.radians(lon) * np.cos(lat0) * R_TERRE, np.radians(lat) * R_TERRE])

    P = xy(stations["lat"].values, stations["lon"].values)
    meilleur = np.full(len(P), np.inf)
    res = np.empty((len(P), 3), dtype=object)
    for axe, a in axes.sort_values(["axe", "pk"]).groupby("axe"):
        Q = xy(a["lat"].values, a["lon"].values)
        A, B = Q[:-1], Q[1:]
        pkA, pkB = a["pk"].values[:-1], a["pk"].values[1:]
        AB = B - A
        L2 = np.maximum((AB ** 2).sum(1), 1e-9)
        for i0 in range(0, len(P), 2000):  # par paquets pour limiter la mémoire
            p = P[i0:i0 + 2000, None, :]
            t = np.clip(((p - A) * AB).sum(-1) / L2, 0, 1)
            proj = A + t[..., None] * AB
            dist = np.sqrt(((p - proj) ** 2).sum(-1))
            j = dist.argmin(1)
            dmin = dist[np.arange(len(j)), j]
            croix = AB[j, 0] * (p[:, 0, 1] - A[j, 1]) - AB[j, 1] * (p[:, 0, 0] - A[j, 0])
            sens = np.where(croix < 0, 1, 2)  # à droite du sens des pk croissants -> sens 1
            pk = pkA[j] + t[np.arange(len(j)), j] * (pkB[j] - pkA[j])
            mieux = dmin < meilleur[i0:i0 + 2000]
            idx = np.arange(i0, i0 + len(j))[mieux]
            meilleur[idx] = dmin[mieux]
            res[idx] = np.column_stack([np.full(mieux.sum(), axe), sens[mieux], np.round(pk[mieux], 2)])
    out = stations.copy()
    out["distance_axe_m"] = np.round(meilleur, 1)
    out[["axe", "sens", "pk"]] = res
    return out[out["distance_axe_m"] <= distance_max_m].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--irve-statique", required=True)
    ap.add_argument("--axes", required=True)
    ap.add_argument("--occupation", help="ne garder que les stations présentes dans cet historique")
    ap.add_argument("--distance-max-m", type=float, default=300.0)
    ap.add_argument("--sortie", default="stations.csv")
    args = ap.parse_args()

    st = stations_irve(pd.read_csv(args.irve_statique, low_memory=False))
    if args.occupation:
        ids = pd.read_csv(args.occupation, usecols=["station_id"])["station_id"].unique()
        manquantes = len(set(ids) - set(st["station_id"]))
        print(f"{len(ids)} stations dans l'historique, {manquantes} absentes du fichier statique")
        st = st[st["station_id"].isin(ids)]
    st = st.dropna(subset=["lat", "lon"])
    out = projeter(st, pd.read_csv(args.axes), args.distance_max_m)
    out.to_csv(args.sortie, index=False)
    print(f"{len(out)} stations à moins de {args.distance_max_m:.0f} m d'un axe -> {args.sortie}")
    print(out.groupby(["axe", "sens"]).size().to_string())


if __name__ == "__main__":
    main()
