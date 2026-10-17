"""
Chargement et mise en forme des données d'entrée.

Schéma attendu (voir README.md, section 1) :
  occupation_bornes.csv  horodatage, station_id, nb_pdc, nb_pdc_occupes, nb_pdc_hors_service
  stations.csv           station_id, lat, lon, axe, sens, pk, nb_pdc, puissance_kw
  comptages.csv          horodatage, compteur_id, debit_vh
  compteurs.csv          compteur_id, lat, lon, axe_rattache, sens, pk_projete, delai_rattachement_min
  calendrier.csv         date, ferie, vacances, debut_vacances, fin_vacances, [bison_fute_couleur]
  aretes.csv (option.)   source, cible, temps_min   -> sinon construit automatiquement (graphe.py)
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Donnees:
    index: pd.DatetimeIndex      # pas de temps réguliers
    stations: pd.DataFrame       # une ligne par station (ordre = colonnes de `occ`)
    compteurs: pd.DataFrame      # une ligne par compteur (ordre = colonnes de `debit`)
    occ: np.ndarray              # [T, S] taux d'occupation (0..1), NaN si inconnu
    debit: np.ndarray            # [T, C] débit véh/h, NaN si trou de mesure
    calendrier: pd.DataFrame     # une ligne par pas de temps, variables calendaires
    aretes: pd.DataFrame | None  # graphe fourni par l'utilisateur, sinon None


def agreger_statuts_pdc(statuts: pd.DataFrame, pas="1h") -> pd.DataFrame:
    """
    Si l'historique est au niveau du point de charge (format IRVE dynamique d'Etalab :
    id_station_itinerance, id_pdc_itinerance, horodatage_derniere_maj, etat_pdc, occupation_pdc),
    on le ramène au format station x pas de temps attendu par le reste du code.

    Chaque statut est valable jusqu'au suivant : on le propage (ffill) sur une grille régulière,
    puis on compte les PDC occupés / hors service par station.
    """
    s = statuts.rename(columns={"id_station_itinerance": "station_id", "id_pdc_itinerance": "pdc_id",
                                "horodatage_derniere_maj": "horodatage"}).copy()
    s["horodatage"] = pd.to_datetime(s["horodatage"], utc=True).dt.tz_convert("Europe/Paris").dt.tz_localize(None)
    s["occupe"] = (s["occupation_pdc"] == "occupe").astype(float)
    s["hs"] = (s["etat_pdc"] == "hors_service").astype(float)
    grille = pd.date_range(s["horodatage"].min().floor(pas), s["horodatage"].max().ceil(pas), freq=pas)
    morceaux = []
    for (station, pdc), g in s.sort_values("horodatage").groupby(["station_id", "pdc_id"]):
        g = g.set_index("horodatage")[["occupe", "hs"]]
        g = g[~g.index.duplicated(keep="last")]
        # moyenne sur le pas de temps : un PDC occupé 30 min sur 1 h compte 0,5
        minutes = pd.date_range(grille[0], grille[-1], freq="1min", inclusive="left")
        g = g.reindex(g.index.union(minutes)).ffill().reindex(minutes)  # statut valable jusqu'au suivant
        fin = g.resample(pas).mean().reindex(grille)
        fin["station_id"] = station
        morceaux.append(fin)
    m = pd.concat(morceaux).rename_axis("horodatage").reset_index()
    m = m[m["horodatage"] < grille[-1]]  # dernier pas incomplet
    out = m.groupby(["horodatage", "station_id"]).agg(nb_pdc=("occupe", "size"), nb_pdc_occupes=("occupe", "sum"),
                                                      nb_pdc_hors_service=("hs", "sum")).reset_index()
    return out


def convertir_snapshots_qualicharge(snap: pd.DataFrame, pas="1h", periode_collecte="5min",
                                    statut_perime="24h", couverture_min=0.5) -> pd.DataFrame:
    """
    Convertit l'export de la collecte Qualicharge (API map.qualicharge.beta.gouv.fr/api/irve/points/,
    une ligne = un état de station : station_key, fetched_at, last_seen_at, latest_status_at,
    availability.*_count) au format `occupation_bornes.csv`.

    Lecture retenue des colonnes (vérifiée sur l'extrait du 9 juillet 2026) :
      - libre + occupé + réservé + inconnu = pdcs_with_dynamic_count (toujours vrai sur l'extrait) ;
      - hors service = pdcs_with_dynamic_count - en_service_count ;
      - taux d'occupation = (occupé + réservé) / (libre + occupé + réservé) : les PDC à l'état
        inconnu sont exclus du dénominateur au lieu d'être comptés libres.

    Chaque ligne est supposée valable de `fetched_at` jusqu'à la ligne suivante de la même station,
    sans dépasser `last_seen_at` + une période de collecte. Un statut dont `latest_status_at` est
    plus vieux que `statut_perime` est considéré comme une remontée morte et devient inconnu.
    Les valeurs sont moyennées au prorata du temps sur chaque pas ; un pas couvert à moins de
    `couverture_min` est abandonné.
    """
    a = "availability."
    d = pd.DataFrame({
        "station_id": snap["station_key"],
        "debut": pd.to_datetime(snap["fetched_at"], utc=True, format="ISO8601"),
        "vu": pd.to_datetime(snap["last_seen_at"], utc=True, format="ISO8601"),
        "statut": pd.to_datetime(snap["latest_status_at"], utc=True, format="ISO8601"),
        "nb_pdc": snap[a + "pdcs_with_dynamic_count"],
        "nb_pdc_occupes": snap[a + "occupied_count"] + snap[a + "reserved_count"],
        "nb_pdc_hors_service": snap[a + "pdcs_with_dynamic_count"] - snap[a + "en_service_count"],
        "nb_pdc_statut_connu": snap[a + "libre_count"] + snap[a + "occupied_count"] + snap[a + "reserved_count"],
    }).sort_values(["station_id", "debut"])
    suivant = d.groupby("station_id")["debut"].shift(-1)
    fin = d["vu"] + pd.Timedelta(periode_collecte)
    fin = fin.where(suivant.isna() | (suivant > fin), suivant)
    d["fin"] = fin.where(fin < d["statut"] + pd.Timedelta(statut_perime), d["statut"] + pd.Timedelta(statut_perime))
    d = d[d["fin"] > d["debut"]]
    for c in ("debut", "fin"):  # heure de Paris, sans fuseau, comme le reste du code
        d[c] = d[c].dt.tz_convert("Europe/Paris").dt.tz_localize(None)

    # découpe chaque intervalle sur les pas de temps qu'il recouvre
    dp = pd.Timedelta(pas)
    d["pas0"] = d["debut"].dt.floor(pas)
    n = ((d["fin"] - d["pas0"]) / dp).apply(np.ceil).astype(int).clip(lower=1)
    e = d.loc[d.index.repeat(n)].copy()
    e["horodatage"] = e["pas0"] + dp * e.groupby(level=0).cumcount().values
    e["w"] = ((e[["fin"]].min(axis=1).clip(upper=e["horodatage"] + dp)
               - e["debut"].clip(lower=e["horodatage"])) / dp).clip(lower=0)
    cols = ["nb_pdc", "nb_pdc_occupes", "nb_pdc_hors_service", "nb_pdc_statut_connu"]
    for c in cols:
        e[c] = e[c] * e["w"]
    g = e.groupby(["horodatage", "station_id"])[cols + ["w"]].sum()
    g = g[g["w"] >= couverture_min]
    out = g[cols].div(g["w"], axis=0).assign(couverture=g["w"].clip(upper=1)).reset_index()
    return out


def charger(dossier: str | Path, pas="1h") -> Donnees:
    d = Path(dossier)
    stations = pd.read_csv(d / "stations.csv")
    compteurs = pd.read_csv(d / "compteurs.csv")
    occ = pd.read_csv(d / "occupation_bornes.csv", parse_dates=["horodatage"])
    cpt = pd.read_csv(d / "comptages.csv", parse_dates=["horodatage"])
    cal = pd.read_csv(d / "calendrier.csv", parse_dates=["date"])
    aretes = pd.read_csv(d / "aretes.csv") if (d / "aretes.csv").exists() else None

    debut = max(occ["horodatage"].min(), cpt["horodatage"].min()).floor(pas)
    fin = min(occ["horodatage"].max(), cpt["horodatage"].max()).floor(pas)
    index = pd.date_range(debut, fin, freq=pas)

    if "nb_pdc_statut_connu" in occ:  # format Qualicharge : les PDC à l'état inconnu sont exclus
        occ["taux"] = (occ["nb_pdc_occupes"] / occ["nb_pdc_statut_connu"].where(occ["nb_pdc_statut_connu"] > 0)).clip(0, 1)
    else:
        dispo = (occ["nb_pdc"] - occ["nb_pdc_hors_service"].fillna(0)).clip(lower=1)
        occ["taux"] = (occ["nb_pdc_occupes"] / dispo).clip(0, 1)
    occ["horodatage"] = occ["horodatage"].dt.floor(pas)
    m_occ = occ.pivot_table(index="horodatage", columns="station_id", values="taux", aggfunc="mean").reindex(index)
    cpt["horodatage"] = cpt["horodatage"].dt.floor(pas)
    m_deb = cpt.pivot_table(index="horodatage", columns="compteur_id", values="debit_vh", aggfunc="mean").reindex(index)

    stations = stations[stations["station_id"].isin(m_occ.columns)].reset_index(drop=True)
    compteurs = compteurs[compteurs["compteur_id"].isin(m_deb.columns)].reset_index(drop=True)
    m_occ = m_occ[stations["station_id"]]
    m_deb = m_deb[compteurs["compteur_id"]]

    calj = cal.set_index("date").reindex(index.normalize()).fillna(0)
    calj.index = index
    calj["heure"] = index.hour + index.minute / 60
    calj["jour_semaine"] = index.dayofweek
    calj["mois"] = index.month
    if "bison_fute_couleur" not in calj:
        calj["bison_fute_couleur"] = 0

    print(f"Données : {len(index)} pas de {pas}, {len(stations)} stations, {len(compteurs)} compteurs, "
          f"occupation manquante {np.isnan(m_occ.values).mean():.1%}, débit manquant {np.isnan(m_deb.values).mean():.1%}")
    return Donnees(index, stations, compteurs, m_occ.values.astype(float), m_deb.values.astype(float), calj, aretes)
