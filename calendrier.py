"""
Variables de calendrier pour les deux étapes : vacances scolaires par zone, jours fériés et ponts,
couleurs Bison Futé par sens. Produit une table journalière (une ligne par date) à joindre aux
données horaires sur la date.

Sources (open data, à télécharger dans donnees_reelles/calendrier/brut/) :
  - vacances : https://www.data.gouv.fr/datasets/vacances-scolaires-par-zones  (CSV : date,
    vacances_zone_a, vacances_zone_b, vacances_zone_c, nom_vacances ; depuis 1995)
  - fériés   : https://calendrier.api.gouv.fr/jours-feries/metropole.json  ({"AAAA-MM-JJ": "nom"}),
    ou un CSV date,nom
  - Bison Futé : CSV (date, sens, zone, couleur) extrait des calendriers annuels PDF ;
    sens = depart|retour, zone = 1..6 ou 0 pour « toute la France », couleur = vert|orange|rouge|noir.

Usage :
    python calendrier.py --vacances brut/vacances.csv --feries brut/jours_feries.json \\
        [--bison-fute bison_fute.csv] --debut 2024-01-01 --fin 2026-12-31 --sortie calendrier.csv
"""
import argparse
import json

import numpy as np
import pandas as pd

NIVEAUX_BF = {"vert": 0, "orange": 1, "rouge": 2, "noir": 3}
ZONES = ["a", "b", "c"]


def vacances(chemin, jours: pd.DatetimeIndex) -> pd.DataFrame:
    v = pd.read_csv(chemin, parse_dates=["date"]).set_index("date").reindex(jours)
    out = pd.DataFrame(index=jours)
    for z in ZONES:
        col = v[f"vacances_zone_{z}"].astype(str).str.lower().isin(["true", "1", "vrai"]).astype(int)
        out[f"vac_{z}"] = col
        # position dans la période : 1er et dernier jour, et jours depuis le début ou avant la fin,
        # pour que le modèle distingue un départ (début) d'un retour (fin)
        bloc = (col != col.shift()).cumsum()
        rang = col.groupby(bloc).cumcount()
        reste = col.groupby(bloc).cumcount(ascending=False)
        out[f"vac_{z}_jour"] = np.where(col == 1, rang, -1)
        out[f"vac_{z}_reste"] = np.where(col == 1, reste, -1)
    out["nb_zones_vac"] = out[[f"vac_{z}" for z in ZONES]].sum(axis=1)
    out["nom_vacances"] = v["nom_vacances"].fillna("").values
    return out


def feries(chemin, jours: pd.DatetimeIndex) -> pd.DataFrame:
    if str(chemin).endswith(".json"):
        f = pd.to_datetime(list(json.load(open(chemin, encoding="utf-8")).keys()))
    else:  # CSV date,nom
        f = pd.to_datetime(pd.read_csv(chemin)["date"])
    out = pd.DataFrame(index=jours)
    out["ferie"] = jours.isin(f).astype(int)
    dow = jours.dayofweek
    # pont : vendredi après un jeudi férié, lundi avant un mardi férié
    out["pont"] = (((dow == 4) & (jours - pd.Timedelta(days=1)).isin(f))
                   | ((dow == 0) & (jours + pd.Timedelta(days=1)).isin(f))).astype(int)
    out["veille_ferie"] = (jours + pd.Timedelta(days=1)).isin(f).astype(int)
    out["lendemain_ferie"] = (jours - pd.Timedelta(days=1)).isin(f).astype(int)
    return out


def bison_fute(chemin, jours: pd.DatetimeIndex) -> pd.DataFrame:
    """Niveau par sens : maximum sur les zones (national), et niveau par zone si fourni."""
    b = pd.read_csv(chemin, parse_dates=["date"])
    b["niveau"] = b["couleur"].str.lower().map(NIVEAUX_BF)
    out = pd.DataFrame(index=jours)
    for sens in ["depart", "retour"]:
        s = b[b["sens"] == sens]
        out[f"bf_{sens}"] = s.groupby("date")["niveau"].max().reindex(jours).fillna(0).astype(int)
        for zone in sorted(s["zone"].unique()):
            if zone == 0:
                continue
            out[f"bf_{sens}_z{zone}"] = (s[s["zone"] == zone].groupby("date")["niveau"].max()
                                         .reindex(jours).fillna(0).astype(int))
    return out


def construire(vac, fer, bf=None, debut="2024-01-01", fin="2026-12-31") -> pd.DataFrame:
    jours = pd.date_range(debut, fin, freq="D")
    parts = [vacances(vac, jours), feries(fer, jours)]
    if bf:
        parts.append(bison_fute(bf, jours))
    cal = pd.concat(parts, axis=1)
    cal.insert(0, "jour_semaine", jours.dayofweek)
    cal.index.name = "date"
    return cal.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vacances", required=True)
    ap.add_argument("--feries", required=True)
    ap.add_argument("--bison-fute")
    ap.add_argument("--debut", default="2024-01-01")
    ap.add_argument("--fin", default="2026-12-31")
    ap.add_argument("--sortie", default="calendrier.csv")
    a = ap.parse_args()
    cal = construire(a.vacances, a.feries, a.bison_fute, a.debut, a.fin)
    cal.to_csv(a.sortie, index=False)
    print(f"{len(cal)} jours, {cal.shape[1] - 1} variables -> {a.sortie}")


if __name__ == "__main__":
    main()
