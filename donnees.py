"""
Chargement et mise en forme des données d'entrée.

Schéma attendu (voir README.md, section 1) :
  occupation_bornes.csv  horodatage, station_id, nb_pdc, nb_pdc_occupes, nb_pdc_hors_service
  stations.csv           station_id, lat, lon, axe, sens, pk, nb_pdc, puissance_kw
  comptages.csv          horodatage, compteur_id, debit_vh                 (Avatar)
  here/here_troncons.csv horodatage, axe, sens, troncon_pk, congestion, jam_factor, [indice_volume]  (HERE, here_flux.py)
  compteurs.csv          compteur_id, lat, lon, axe_rattache, sens, pk_projete, delai_rattachement_min
  calendrier.csv         date + variables numériques du jour (calendrier.py : vac_a.., ferie, pont, bf_depart..)
  aretes.csv (option.)   source, cible, temps_min   -> sinon construit automatiquement (graphe.py)
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# variables de trafic exprimées en volume : normalisées par la moyenne du nœud (1 = trafic moyen) ;
# les autres (congestion, jam factor) sont déjà sans unité, entre 0 et 1
VARIABLES_VOLUME = ("debit", "volume_here")


@dataclass
class Donnees:
    index: pd.DatetimeIndex      # pas de temps réguliers
    stations: pd.DataFrame       # une ligne par station (ordre = colonnes de `occ`)
    compteurs: pd.DataFrame      # une ligne par nœud de trafic, compteur Avatar ou tronçon HERE (colonne `source`)
    occ: np.ndarray              # [T, S] taux d'occupation (0..1), NaN si inconnu
    trafic: np.ndarray           # [T, C, V] variables de trafic par nœud, NaN si absente ou trou de mesure
    variables_trafic: list       # noms des V variables : debit (Avatar), congestion, jam_factor, volume_here (HERE)
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


def _trouver(d: Path, *candidats):
    """Premier fichier existant, en clair ou compressé (.xz, lu tel quel par pandas) : la disposition du
    jeu synthétique (tout à la racine) ou celle de donnees_reelles/ (debits/, calendrier/)."""
    for c in candidats:
        for nom in (c, c + ".xz"):
            if (Path(d) / nom).exists():
                return Path(d) / nom
    raise FileNotFoundError(f"aucun de {candidats} (ni .xz) dans {d}")


def lire_comptages(dossier, **kw) -> pd.DataFrame:
    """Comptages Avatar : comptages.csv s'il existe, sinon les morceaux compressés comptages_*.csv.xz
    (découpés par période pour tenir sous les limites de transfert), lus et mis bout à bout."""
    dossier = Path(dossier)
    if (dossier / "comptages.csv").exists():
        return pd.read_csv(dossier / "comptages.csv", **kw)
    if (dossier / "comptages.csv.xz").exists():
        return pd.read_csv(dossier / "comptages.csv.xz", **kw)
    morceaux = sorted(dossier.glob("comptages_*.csv*"))
    if not morceaux:
        raise FileNotFoundError(f"ni comptages.csv ni comptages_*.csv.xz dans {dossier}")
    return pd.concat([pd.read_csv(f, **kw) for f in morceaux], ignore_index=True)


def charger_avatar(d: Path, pas, fichier_compteurs=None):
    """Compteurs Avatar : une variable `debit` (véh/h). d : dossier qui contient compteurs.csv et les
    comptages, ou son parent (donnees_reelles/, avec debits/)."""
    d = Path(d)
    if not list(d.glob("comptages*.csv*")) and (d / "debits").exists():
        d = d / "debits"
    compteurs = pd.read_csv(fichier_compteurs or _trouver(d, "compteurs.csv"))
    cpt = lire_comptages(d,
                         usecols=["horodatage", "compteur_id", "debit_vh"], parse_dates=["horodatage"])
    cpt = cpt[cpt["compteur_id"].isin(compteurs["compteur_id"])]
    cpt["horodatage"] = cpt["horodatage"].dt.floor(pas)
    m = cpt.pivot_table(index="horodatage", columns="compteur_id", values="debit_vh", aggfunc="mean")
    compteurs = compteurs[compteurs["compteur_id"].isin(m.columns)].reset_index(drop=True)
    noeuds = pd.DataFrame({"compteur_id": compteurs["compteur_id"], "axe_rattache": compteurs["axe_rattache"],
                           "sens": compteurs["sens"], "pk_projete": compteurs["pk_projete"],
                           "delai_rattachement_min": compteurs.get("delai_rattachement_min", 10.0),
                           "source": "avatar"})
    return noeuds, {"debit": m[noeuds["compteur_id"]]}


def charger_here(chemin, pas, longueur_troncon_km=10.0):
    """
    Tronçons HERE au format de `here_flux.py` (here_troncons.csv) : horodatage, axe, sens, troncon_pk,
    vitesse_kmh, vitesse_fluide_kmh, congestion, jam_factor, confiance, et en option `indice_volume`
    ou `nb_sondes` (historique HERE : nombre de véhicules sondes, indice relatif de volume).
    Chaque tronçon (axe, sens, tranche de 10 km) devient un nœud de trafic posé sur l'axe, au milieu
    de sa tranche, sans délai de rattachement. Les relevés (toutes les 5 à 15 min) sont moyennés au pas.
    """
    h = pd.read_csv(chemin, parse_dates=["horodatage"])
    h["noeud"] = "HERE_" + h["axe"] + "_" + h["sens"].astype(int).astype(str) + "_" + h["troncon_pk"].astype(int).astype(str)
    h["horodatage"] = h["horodatage"].dt.floor(pas)
    h["jam_factor"] = h["jam_factor"] / 10  # 0..10 chez HERE
    vol = next((c for c in ("indice_volume", "nb_sondes") if c in h), None)
    if vol:
        h["volume_here"] = h[vol]
    variables = ["congestion", "jam_factor"] + (["volume_here"] if vol else [])
    mats = {v: h.pivot_table(index="horodatage", columns="noeud", values=v, aggfunc="mean") for v in variables}
    t = h.drop_duplicates("noeud").set_index("noeud").loc[mats["congestion"].columns]
    noeuds = pd.DataFrame({"compteur_id": t.index, "axe_rattache": t["axe"].values, "sens": t["sens"].astype(int).values,
                           "pk_projete": t["troncon_pk"].values + longueur_troncon_km / 2,
                           "delai_rattachement_min": 0.0, "source": "here"})
    return noeuds, {v: m[noeuds["compteur_id"]] for v, m in mats.items()}


def charger(dossier: str | Path, pas="1h", trafic="avatar", here=None, fichier_compteurs=None,
            avatar=None, occupation=None) -> Donnees:
    """trafic : "avatar", "here" ou "here+avatar" ; here : chemin de here_troncons.csv
    (par défaut <dossier>/here/here_troncons.csv) ; fichier_compteurs : liste de compteurs à garder
    (par ex. rattachement_interurbains/compteurs_retenus.csv) ; avatar : dossier des comptages Avatar
    s'il n'est pas dans <dossier> (ex. data_IRVE/avatar) ; occupation : fichier d'occupation s'il n'est
    pas <dossier>/occupation_bornes.csv[.xz]. Tous les CSV peuvent être compressés en .xz."""
    d = Path(dossier)
    stations = pd.read_csv(_trouver(d, "stations.csv"))
    occ = pd.read_csv(occupation or _trouver(d, "occupation_bornes.csv"), parse_dates=["horodatage"])
    cal = pd.read_csv(_trouver(d, "calendrier.csv", "calendrier/calendrier.csv"), parse_dates=["date"])
    aretes = pd.read_csv(d / "aretes.csv") if (d / "aretes.csv").exists() else None  # graphe fourni

    sources = []
    if "here" in trafic:
        sources.append(charger_here(here or _trouver(d / "here", "here_troncons.csv"), pas))
    if "avatar" in trafic:
        sources.append(charger_avatar(avatar or d, pas, fichier_compteurs))
    if not sources:
        raise ValueError(f"trafic inconnu : {trafic}")
    noeuds = pd.concat([n for n, _ in sources], ignore_index=True)
    debut_t = max(min(m.index.min() for m in mats.values()) for _, mats in sources)
    fin_t = min(max(m.index.max() for m in mats.values()) for _, mats in sources)

    debut = max(occ["horodatage"].min(), debut_t).floor(pas)
    fin = min(occ["horodatage"].max(), fin_t).floor(pas)
    index = pd.date_range(debut, fin, freq=pas)

    if "nb_pdc_statut_connu" in occ:  # format Qualicharge : les PDC à l'état inconnu sont exclus
        occ["taux"] = (occ["nb_pdc_occupes"] / occ["nb_pdc_statut_connu"].where(occ["nb_pdc_statut_connu"] > 0)).clip(0, 1)
    else:
        dispo = (occ["nb_pdc"] - occ["nb_pdc_hors_service"].fillna(0)).clip(lower=1)
        occ["taux"] = (occ["nb_pdc_occupes"] / dispo).clip(0, 1)
    occ["horodatage"] = occ["horodatage"].dt.floor(pas)
    m_occ = occ.pivot_table(index="horodatage", columns="station_id", values="taux", aggfunc="mean").reindex(index)
    stations = stations[stations["station_id"].isin(m_occ.columns)].reset_index(drop=True)
    m_occ = m_occ[stations["station_id"]]

    # [T, C, V] : un nœud Avatar n'a que `debit`, un tronçon HERE n'a que ses variables ; le reste est NaN
    variables = list(dict.fromkeys(v for _, mats in sources for v in mats))
    tr = np.full((len(index), len(noeuds), len(variables)), np.nan, dtype=np.float32)
    col = 0
    for n, mats in sources:
        for v, m in mats.items():
            tr[:, col:col + len(n), variables.index(v)] = m.reindex(index).values
        col += len(n)

    calj = cal.set_index("date").reindex(index.normalize()).fillna(0)
    calj.index = index
    calj["heure"] = index.hour + index.minute / 60
    calj["jour_semaine"] = index.dayofweek
    calj["mois"] = index.month

    manque = {v: f"{np.isnan(tr[..., k]).mean():.1%}" for k, v in enumerate(variables)}
    print(f"Données : {len(index)} pas de {pas}, {len(stations)} stations, {len(noeuds)} nœuds de trafic "
          f"({noeuds['source'].value_counts().to_dict()}), occupation manquante {np.isnan(m_occ.values).mean():.1%}, "
          f"trafic manquant {manque}")
    return Donnees(index, stations, noeuds, m_occ.values.astype(float), tr, variables, calj, aretes)
