"""
Calcule (ou recharge depuis un cache) le taux d'occupation « calculé » par le GNN, sur toute la
période couverte par les données — pas seulement la période de test de `entrainer.py`. Réutilise
les modèles déjà entraînés (resultats_reels/), sans les ré-ajuster : même logique que
`entrainer.py`, mais appelée en inférence.

`preparer_contexte()` fait la préparation lourde une fois (chargement, graphe, trafic observé,
modèle GNN chargé) et renvoie un dict réutilisé pour :
  - `calculer_complet(ctx)` : toute la période, mis en cache dans resultats_reels/calcule_complet.csv.xz
    (recalculé seulement si ce fichier est plus vieux que les modèles sauvegardés) ;
  - `calculer_scenario(ctx, date, niveaux_zone)` : un jour historique donné, avec les couleurs
    Bison Futé remplacées par un scénario hypothétique (volet prédiction / what-if). Limité aux
    jours où l'historique réel suffit (mêmes contraintes de retard que le calcul complet) : on ne
    change que l'entrée calendaire du GNN, le trafic et l'occupation passée restent les vraies
    valeurs observées ce jour-là.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import graphe  # noqa: E402
import entrainer as E  # noqa: E402
from donnees import charger, VARIABLES_VOLUME  # noqa: E402
from gnn import STGNNLeger, predire_gnn  # noqa: E402
from scenarios import charge_depuis_occupation, occupation_depuis_charge  # noqa: E402

PENETRATION_REF = 0.03  # part de VE dans le trafic de l'historique (scenarios.py)

DONNEES_DIR = ROOT / "donnees_reelles"
AVATAR_DIR = ROOT.parent / "inputs" / "avatar"
COMPTEURS_FILE = DONNEES_DIR / "rattachement_interurbains" / "compteurs_retenus.csv"
RESULTATS_DIR = ROOT / "resultats_reels"
CACHE_FILE = RESULTATS_DIR / "calcule_complet.csv.xz"

# noms des colonnes calendaires Bison Futé que le volet prédiction peut surcharger
NOMS_BF = ["bf_depart", "bf_retour"] + [f"bf_depart_z{z}" for z in range(1, 7)] + [f"bf_retour_z{z}" for z in range(1, 7)]


def preparer_contexte():
    """Chargement + inférence préparatoire (une fois). Ne ré-entraîne rien : recharge les modèles
    de resultats_reels/ et reconstruit exactement les tenseurs que entrainer.py utilisait."""
    cfg = json.load(open(RESULTATS_DIR / "config.json", encoding="utf-8"))
    pas, H, lags = cfg["pas"], cfg["horizon_pas"], cfg["lags"]

    d = charger(str(DONNEES_DIR), pas, trafic=cfg["trafic"], fichier_compteurs=str(COMPTEURS_FILE), avatar=str(AVATAR_DIR))
    T, S, C = len(d.index), len(d.stations), len(d.compteurs)
    V = len(d.variables_trafic)
    pas_min = pd.Timedelta(pas) / pd.Timedelta(minutes=1)
    pas_par_jour = int(round(24 * 60 / pas_min))

    noeuds = graphe.table_noeuds(d.stations, d.compteurs)
    aretes = graphe.construire_aretes(noeuds)
    temps = graphe.matrice_temps(noeuds, aretes)
    temps_cs, temps_sc = temps[S:, :S], temps[:S, S:]
    A_amont, A_aval = graphe.adjacences(temps)

    trafic_n = d.trafic.copy()
    for k, v in enumerate(d.variables_trafic):
        if v in VARIABLES_VOLUME:
            moy_map = cfg["moyennes_normalisation"][v]
            moy = np.array([moy_map.get(cid, np.nan) for cid in d.compteurs["compteur_id"]])
            trafic_n[..., k] = d.trafic[..., k] / np.where(moy > 0, moy, np.nan)

    Xcal, noms_cal = E.variables_calendaires(d.calendrier)

    hgb_dump = joblib.load(RESULTATS_DIR / "modeles_hgb.joblib")
    etape1 = hgb_dump["etape1"]

    axe_code = pd.factorize(d.compteurs["axe_rattache"])[0]
    stat1 = np.column_stack([np.arange(C), d.compteurs["pk_projete"].values, d.compteurs["sens"].values, axe_code]).astype(np.float32)

    trafic_prevu_n = np.full_like(trafic_n, np.nan)
    for k, v in enumerate(d.variables_trafic):
        x = trafic_n[..., k]
        info = etape1[v]
        mdl, noms1 = info["modele"], info["noms"]
        id_to_idx = {cid: i for i, cid in enumerate(d.compteurs["compteur_id"])}
        cols = np.array([id_to_idx[cid] for cid in info["noeuds"]])
        x_cols = x[:, cols]
        lags_v = [graphe.decaler(x_cols, L) for L in lags]
        moy24 = E.moyenne_glissante_decalee(x_cols, pas_par_jour, H)

        def table1(ii, lags_v=lags_v, moy24=moy24, cols=cols):
            n = len(ii)
            blocs = [np.repeat(Xcal[ii, None, :], len(cols), 1), np.repeat(stat1[None, cols], n, 0)]
            blocs += [l[ii, :, None] for l in lags_v] + [moy24[ii, :, None]]
            return np.concatenate(blocs, -1).astype(np.float32)

        prev = np.concatenate([mdl.predict(table1(np.arange(T)[a:a + 256]).reshape(-1, len(noms1))).reshape(-1, len(cols))
                               for a in range(0, T, 256)])
        trafic_prevu_n[:, cols, k] = prev
    trafic_obs_n = np.where(np.isnan(trafic_n), trafic_prevu_n, trafic_n)

    occ = d.occ
    occ_lags = {k: graphe.decaler(occ, k) for k in lags}
    occ_moy_jour = E.moyenne_glissante_decalee(occ, pas_par_jour, H)

    st = d.stations

    gnn_dump = torch.load(RESULTATS_DIR / "modele_gnn.pt", map_location="cpu")
    n_feat, n_cal_saved, T_in, adaptative = gnn_dump["n_feat"], gnn_dump["n_cal"], gnn_dump["T_in"], gnn_dump["adaptative"]
    N = S + C
    types = d.compteurs.get("source", pd.Series("avatar", index=d.compteurs.index)).values

    def tenseur_gnn(tr_n):
        f_tr = np.zeros((T, N, V), dtype=np.float32); f_tr[:, S:] = np.nan_to_num(tr_n)
        f_type = np.zeros((T, N, 3), dtype=np.float32)
        f_type[:, :S, 0] = 1; f_type[:, S:, 1] = types == "avatar"; f_type[:, S:, 2] = types == "here"
        f_occ = np.zeros((T, N, len(lags) + 1), dtype=np.float32); m_occ = np.zeros((T, N), dtype=np.float32)
        for j, k in enumerate(lags):
            f_occ[:, :S, j] = np.nan_to_num(occ_lags[k], nan=0)
        f_occ[:, :S, -1] = np.nan_to_num(occ_moy_jour)
        m_occ[:, :S] = ~np.isnan(occ_lags[H])
        cap = np.zeros((T, N), dtype=np.float32); cap[:, :S] = st["nb_pdc"].values / 10
        return np.concatenate([f_tr, f_type, f_occ, m_occ[..., None], cap[..., None]], -1)

    n_tr = int(T * 0.70)
    i_tr = np.arange(T)[:n_tr]
    noms_cal_sel = [n for n in noms_cal if n not in ("heure", "jour_semaine", "mois")]
    cal_sel = Xcal[:, [noms_cal.index(n) for n in noms_cal_sel]]
    normaliseur = np.maximum(1, np.abs(cal_sel[i_tr]).max(0)).astype(np.float32)
    cal_g = (cal_sel / normaliseur).astype(np.float32)

    G_obs = tenseur_gnn(trafic_obs_n)
    assert G_obs.shape[-1] == n_feat and cal_g.shape[1] == n_cal_saved

    modele = STGNNLeger(N, n_feat, A_amont, A_aval, n_cal=n_cal_saved, adaptative=adaptative)
    modele.load_state_dict(gnn_dump["etat"])
    modele.eval()

    # même garde que entrainer.py : pas de fenêtre GNN valide avant T_in-1 ni avant le plus grand retard
    debut = max(T_in - 1, max(lags))

    return {
        "d": d, "T": T, "S": S, "lags": lags, "H": H, "T_in": T_in, "debut": debut,
        "noms_cal_sel": noms_cal_sel, "normaliseur": normaliseur, "cal_g": cal_g,
        "G_obs": G_obs, "modele": modele, "occ": occ,
    }


def calculer_complet(ctx):
    """Renvoie un DataFrame long (station_id, date, heure, calcule) sur tous les pas de temps où
    le GNN a assez d'historique (à partir de ctx['debut'])."""
    d, S, T_in, T, debut = ctx["d"], ctx["S"], ctx["T_in"], ctx["T"], ctx["debut"]
    idx_valides = np.arange(debut, T)
    p_gnn = predire_gnn(ctx["modele"], ctx["G_obs"], idx_valides, T_in, S, cal=ctx["cal_g"])

    calcule = np.full((T, S), np.nan, dtype=np.float32)
    calcule[idx_valides] = p_gnn

    dates = d.index.strftime("%Y-%m-%d").values
    heures = d.index.hour.values
    st = d.stations
    lignes = []
    for c, sid in enumerate(st["station_id"]):
        col = calcule[:, c]
        m = ~np.isnan(col)
        if not m.any():
            continue
        lignes.append(pd.DataFrame({"station_id": sid, "date": dates[m], "heure": heures[m], "calcule": col[m]}))
    return pd.concat(lignes, ignore_index=True)


def _cache_a_jour():
    if not CACHE_FILE.exists():
        return False
    t_cache = CACHE_FILE.stat().st_mtime
    sources = [RESULTATS_DIR / "modele_gnn.pt", RESULTATS_DIR / "modeles_hgb.joblib", RESULTATS_DIR / "config.json"]
    return all(s.exists() and s.stat().st_mtime <= t_cache for s in sources)


def obtenir_calcule_complet(ctx):
    """DataFrame pivot (station_id, date) -> colonnes 0..23 (taux calculé par le GNN), ou NaN
    aux pas de temps trop proches du début de la période (pas assez d'historique)."""
    if _cache_a_jour():
        long = pd.read_csv(CACHE_FILE)
    else:
        print("Calcul des prévisions GNN sur toute la période (pas de cache à jour)...", flush=True)
        long = calculer_complet(ctx)
        RESULTATS_DIR.mkdir(parents=True, exist_ok=True)
        long.to_csv(CACHE_FILE, index=False)
        print(f"-> {CACHE_FILE} ({len(long)} lignes)", flush=True)
    pivot = long.pivot_table(index=["station_id", "date"], columns="heure", values="calcule")
    return pivot.reindex(columns=range(24))


def dates_calculables(ctx):
    """Dates (YYYY-MM-DD) où le GNN a assez d'historique réel pour tourner — le volet prédiction
    ne doit proposer que celles-ci."""
    d, debut, T = ctx["d"], ctx["debut"], ctx["T"]
    return set(d.index[debut:T].strftime("%Y-%m-%d").unique())


def calculer_scenario(ctx, date, niveaux_zone, penetration=None):
    """Rejoue le GNN sur un jour historique en remplaçant ses couleurs Bison Futé par un scénario
    hypothétique (niveaux_zone : {1: 0..3, ..., 6: 0..3}). Le trafic et l'occupation passée
    restent les vraies valeurs observées : seule l'entrée calendaire change.

    `penetration` (part de VE dans le trafic, ex. 0.10 pour 10 %) rescale ensuite ce taux via la
    théorie des files d'attente (scenarios.py, Erlang B) : le GNN apprend la demande à la
    pénétration de référence (~3 %, quasi constante sur l'historique, donc inapprenable comme
    variable) ; un changement de pénétration est un effet de charge appliqué après coup, pas une
    entrée du modèle. None ou égal à PENETRATION_REF -> pas de changement.

    Renvoie {station_id: {"reel_pdc": [...24...], "scenario_pdc": [...24...], "nb_pdc": int}},
    ou None si la date n'a pas assez d'historique réel.
    """
    d, S, T_in, debut = ctx["d"], ctx["S"], ctx["T_in"], ctx["debut"]
    idx_jour = np.where(d.index.normalize() == pd.Timestamp(date))[0]
    if len(idx_jour) == 0 or idx_jour.min() < debut:
        return None

    noms_cal_sel, normaliseur = ctx["noms_cal_sel"], ctx["normaliseur"]
    national = max(niveaux_zone.values())
    cal_g_scenario = ctx["cal_g"].copy()
    for z, niveau in niveaux_zone.items():
        for prefixe in ("bf_depart_z", "bf_retour_z"):
            pos = noms_cal_sel.index(f"{prefixe}{z}")
            cal_g_scenario[np.ix_(idx_jour, [pos])] = niveau / normaliseur[pos]
    for nom in ("bf_depart", "bf_retour"):
        pos = noms_cal_sel.index(nom)
        cal_g_scenario[np.ix_(idx_jour, [pos])] = national / normaliseur[pos]

    p_gnn = predire_gnn(ctx["modele"], ctx["G_obs"], idx_jour, T_in, S, cal=cal_g_scenario)  # [len(idx_jour), S]

    st = d.stations
    nb_pdc_tous = np.maximum(np.round(st["nb_pdc"].values).astype(int), 1)
    if penetration and abs(penetration - PENETRATION_REF) > 1e-9:
        charge = charge_depuis_occupation(np.clip(p_gnn, 0, 1), nb_pdc_tous[None, :])
        p_gnn = occupation_depuis_charge(charge * penetration / PENETRATION_REF, nb_pdc_tous[None, :])

    heures = d.index[idx_jour].hour
    out = {}
    for c, sid in enumerate(st["station_id"]):
        nb_pdc = int(nb_pdc_tous[c])
        reel = np.full(24, np.nan, dtype=np.float32)
        scenario = np.full(24, np.nan, dtype=np.float32)
        reel[heures] = ctx["occ"][idx_jour, c]
        scenario[heures] = p_gnn[:, c]
        out[sid] = {
            "reel_pdc": [None if np.isnan(v) else round(float(v) * nb_pdc, 1) for v in reel],
            "scenario_pdc": [None if np.isnan(v) else round(float(v) * nb_pdc, 1) for v in scenario],
            "nb_pdc": nb_pdc,
        }
    return out
