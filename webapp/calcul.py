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
import warnings
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
CALENDRIER_FILE = DONNEES_DIR / "calendrier" / "calendrier.csv"
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
        "G_obs": G_obs, "modele": modele, "occ": occ, "pas_par_jour": pas_par_jour,
        "etape1": etape1, "stat1": stat1, "trafic_obs_n": trafic_obs_n,
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


def bornes_jour_complet(ctx):
    """(premier jour simulable, dernier jour simulable) pour le volet « jour complet » : juste
    après la fin de l'historique réel, jusqu'à la fin du calendrier connu (Bison Futé officiel
    publié à l'avance)."""
    dernier_reel = ctx["d"].index[-1].normalize()
    cal_brut = pd.read_csv(CALENDRIER_FILE, parse_dates=["date"])
    return dernier_reel + pd.Timedelta(days=1), cal_brut["date"].max().normalize()


def _simuler_jour(ctx, t0, trafic_hist, occ_hist, G_hist, cal_g_complet, Xcal_jour, id_to_idx):
    """Calcule un jour (pas_par_jour pas) à l'indice global t0 : prévision du trafic (étape 1,
    HGB) puis de l'occupation (étape 2, GNN), à partir de l'historique déjà connu avant t0 (réel ou
    déjà simulé dans la cascade). `trafic_hist`/`occ_hist`/`G_hist` couvrent les indices [0, t0) et
    s'indexent donc directement comme les indices globaux (pas de décalage à appliquer).
    `Xcal_jour` : calendrier brut (non normalisé) de ce jour, pour l'étape 1 ; `cal_g_complet` :
    calendrier GNN normalisé sur [0, t0 + pas_par_jour), ce jour compris (officiel ou scénario).
    Renvoie (trafic_jour [pas_par_jour, C, V] normalisé, occupation_jour [pas_par_jour, S])."""
    d, S, lags, H, T_in = ctx["d"], ctx["S"], ctx["lags"], ctx["H"], ctx["T_in"]
    pas_par_jour = ctx["pas_par_jour"]
    idx = np.arange(t0, t0 + pas_par_jour)
    n = len(idx)
    C, V = trafic_hist.shape[1], trafic_hist.shape[2]

    trafic_jour = np.zeros((n, C, V), dtype=np.float32)
    for k, v in enumerate(d.variables_trafic):
        info = ctx["etape1"][v]
        mdl, noms1 = info["modele"], info["noms"]
        cols = np.array([id_to_idx[cid] for cid in info["noeuds"]])
        x_cols = trafic_hist[:, cols, k]
        lags_v = [x_cols[idx - L] for L in lags]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # colonnes jamais mesurées -> NaN partout
            moy24 = np.nanmean(np.stack([x_cols[idx - H - i] for i in range(pas_par_jour)], 0), axis=0)
        table = np.concatenate([
            np.repeat(Xcal_jour[:, None, :], len(cols), 1),
            np.repeat(ctx["stat1"][None, cols], n, 0),
            *[l[:, :, None] for l in lags_v], moy24[:, :, None],
        ], -1).astype(np.float32)
        trafic_jour[:, cols, k] = mdl.predict(table.reshape(-1, len(noms1))).reshape(n, len(cols))

    G_jour = _tenseur_jour(ctx, trafic_jour, occ_hist, idx)
    G_complet = np.concatenate([G_hist, G_jour])
    p = predire_gnn(ctx["modele"], G_complet, idx, T_in, S, cal=cal_g_complet)
    return trafic_jour, p


def _tenseur_jour(ctx, trafic_jour, occ_hist, idx):
    """Même format que le tenseur GNN construit dans `preparer_contexte`, mais pour un seul jour
    (idx : indices globaux de ce jour). `occ_hist` couvre [0, idx[0]) (indexation directe)."""
    d, S, lags, H = ctx["d"], ctx["S"], ctx["lags"], ctx["H"]
    pas_par_jour = ctx["pas_par_jour"]
    C, V, N = len(d.compteurs), trafic_jour.shape[-1], ctx["S"] + len(d.compteurs)
    n = len(idx)

    f_tr = np.zeros((n, N, V), dtype=np.float32)
    f_tr[:, S:] = np.nan_to_num(trafic_jour)

    types = d.compteurs.get("source", pd.Series("avatar", index=d.compteurs.index)).values
    f_type = np.zeros((n, N, 3), dtype=np.float32)
    f_type[:, :S, 0] = 1; f_type[:, S:, 1] = types == "avatar"; f_type[:, S:, 2] = types == "here"

    f_occ = np.zeros((n, N, len(lags) + 1), dtype=np.float32)
    m_occ = np.zeros((n, N), dtype=np.float32)
    for j, k in enumerate(lags):
        f_occ[:, :S, j] = np.nan_to_num(occ_hist[idx - k], nan=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # stations sans aucune mesure sur la fenêtre -> NaN
        moy_jour = np.nanmean(np.stack([occ_hist[idx - H - i] for i in range(pas_par_jour)], 0), axis=0)
    f_occ[:, :S, -1] = np.nan_to_num(moy_jour, nan=0)
    m_occ[:, :S] = ~np.isnan(occ_hist[idx - H])

    cap = np.zeros((n, N), dtype=np.float32)
    cap[:, :S] = d.stations["nb_pdc"].values / 10

    return np.concatenate([f_tr, f_type, f_occ, m_occ[..., None], cap[..., None]], -1)


def calculer_jour_complet(ctx, date_cible, niveaux_zone=None, penetration=None):
    """Prédiction complète d'un jour au-delà de l'historique réel (ex. 1ᵉʳ novembre à venir) :
    pas de trafic ni d'occupation mesurés ce jour-là, donc on les prévoit aussi, en cascade jour
    par jour depuis la fin des données réelles jusqu'à `date_cible`, en réinjectant à chaque étape
    les sorties du modèle (trafic prévu, occupation prévue) comme nouvel historique pour le jour
    suivant. Le calendrier des jours intermédiaires est le Bison Futé officiel (déjà publié dans
    calendrier.csv). Seul le dernier jour (`date_cible`) peut en plus recevoir un scénario Bison
    Futé hypothétique (`niveaux_zone`, comme `calculer_scenario`) et une pénétration VE différente
    de la référence historique : dans ce cas ce dernier jour est rejoué une seconde fois avec le
    calendrier modifié, sans toucher au reste de la cascade (identique dans les deux cas).

    Renvoie None si `date_cible` est déjà couverte par l'historique réel (-> `calculer_scenario`)
    ou hors de la plage du calendrier connu, sinon :
      {"heures": [...], "date_cible": "YYYY-MM-DD", "dates_cascade": [...],
       "par_station": {station_id: {"prevu_pdc": [...24...], "scenario_pdc": [...24...] ou None,
       "nb_pdc": int}}}
    """
    d, S, H, T_in = ctx["d"], ctx["S"], ctx["H"], ctx["T_in"]
    pas_par_jour = ctx["pas_par_jour"]
    T0 = ctx["T"]
    dernier_jour = d.index[-1].normalize()
    cible = pd.Timestamp(date_cible).normalize()
    if cible <= dernier_jour:
        return None

    cal_brut = pd.read_csv(CALENDRIER_FILE, parse_dates=["date"]).set_index("date")
    if cible not in cal_brut.index:
        return None

    jours = pd.date_range(dernier_jour + pd.Timedelta(days=1), cible, freq="D")
    n_jours = len(jours)
    pas = d.index[1] - d.index[0]
    index_h = pd.date_range(jours[0], jours[-1] + pd.Timedelta(days=1) - pas, freq=pas)
    calj = cal_brut.reindex(index_h.normalize()).fillna(0)
    calj.index = index_h
    calj["heure"] = index_h.hour + index_h.minute / 60
    calj["jour_semaine"] = index_h.dayofweek
    calj["mois"] = index_h.month
    Xcal_c, noms_cal_c = E.variables_calendaires(calj)
    idx_sel = [noms_cal_c.index(nn) for nn in ctx["noms_cal_sel"]]
    cal_g_c = (Xcal_c[:, idx_sel] / ctx["normaliseur"]).astype(np.float32)

    C, V = len(d.compteurs), len(d.variables_trafic)
    N = S + C
    pas_total = n_jours * pas_par_jour
    trafic_n_ext = np.concatenate([ctx["trafic_obs_n"], np.zeros((pas_total, C, V), dtype=np.float32)])
    occ_ext = np.concatenate([ctx["occ"], np.zeros((pas_total, S), dtype=np.float32)])
    G_ext = np.concatenate([ctx["G_obs"], np.zeros((pas_total, N, ctx["G_obs"].shape[2]), dtype=np.float32)])
    cal_g_ext = np.concatenate([ctx["cal_g"], cal_g_c])

    id_to_idx = {cid: i for i, cid in enumerate(d.compteurs["compteur_id"])}
    t0 = T0
    for j in range(n_jours - 1):  # tous les jours de la cascade SAUF le jour cible (calendrier officiel)
        i0, i1 = j * pas_par_jour, (j + 1) * pas_par_jour
        trafic_jour, occ_jour = _simuler_jour(
            ctx, t0, trafic_n_ext[:t0], occ_ext[:t0], G_ext[:t0],
            cal_g_ext[:t0 + pas_par_jour], Xcal_c[i0:i1], id_to_idx)
        trafic_n_ext[t0:t0 + pas_par_jour] = trafic_jour
        occ_ext[t0:t0 + pas_par_jour] = occ_jour
        G_ext[t0:t0 + pas_par_jour] = _tenseur_jour(ctx, trafic_jour, occ_ext[:t0], np.arange(t0, t0 + pas_par_jour))
        t0 += pas_par_jour

    i0 = (n_jours - 1) * pas_par_jour
    _, occ_officiel = _simuler_jour(
        ctx, t0, trafic_n_ext[:t0], occ_ext[:t0], G_ext[:t0],
        cal_g_ext[:t0 + pas_par_jour], Xcal_c[i0:i0 + pas_par_jour], id_to_idx)

    nb_pdc_tous = np.maximum(np.round(d.stations["nb_pdc"].values).astype(int), 1)
    occ_scenario = None
    scenario_demande = bool(niveaux_zone) or (penetration is not None and abs(penetration - PENETRATION_REF) > 1e-9)
    if scenario_demande:
        calj_scenario = calj.iloc[i0:i0 + pas_par_jour].copy()
        if niveaux_zone:
            national = max(niveaux_zone.values())
            for z, niveau in niveaux_zone.items():
                calj_scenario[f"bf_depart_z{z}"] = niveau
                calj_scenario[f"bf_retour_z{z}"] = niveau
            calj_scenario["bf_depart"] = national
            calj_scenario["bf_retour"] = national
        Xcal_cible_sc, _ = E.variables_calendaires(calj_scenario)
        cal_g_cible_sc = (Xcal_cible_sc[:, idx_sel] / ctx["normaliseur"]).astype(np.float32)
        cal_g_avec_scenario = np.concatenate([cal_g_ext[:t0], cal_g_cible_sc])
        _, occ_scenario = _simuler_jour(
            ctx, t0, trafic_n_ext[:t0], occ_ext[:t0], G_ext[:t0],
            cal_g_avec_scenario, Xcal_cible_sc, id_to_idx)
        if penetration and abs(penetration - PENETRATION_REF) > 1e-9:
            charge = charge_depuis_occupation(np.clip(occ_scenario, 0, 1), nb_pdc_tous[None, :])
            occ_scenario = occupation_depuis_charge(charge * penetration / PENETRATION_REF, nb_pdc_tous[None, :])

    out = {}
    for c, sid in enumerate(d.stations["station_id"]):
        nb_pdc = int(nb_pdc_tous[c])
        out[sid] = {
            "prevu_pdc": [round(float(v) * nb_pdc, 1) for v in occ_officiel[:, c]],
            "scenario_pdc": ([round(float(v) * nb_pdc, 1) for v in occ_scenario[:, c]] if occ_scenario is not None else None),
            "nb_pdc": nb_pdc,
        }
    return {
        "heures": list(range(pas_par_jour)),
        "date_cible": str(cible.date()),
        "dates_cascade": [str(x.date()) for x in jours],
        "par_station": out,
    }
