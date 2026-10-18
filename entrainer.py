"""
Script d'apprentissage du "Bison Futé des bornes".

Chaîne en deux étapes (voir README.md) :
  Étape 1 : prévoir le trafic aux noeuds de trafic à l'horizon H (calendrier + historique du noeud) :
            débit des compteurs Avatar et/ou congestion, jam factor et indice de volume des tronçons HERE.
  Étape 2 : estimer l'occupation des stations à partir du trafic (propagé le long du graphe),
            du calendrier et de l'occupation connue il y a au moins H.

Deux modèles comparés pour l'étape 2 :
  - HGB : gradient boosting (scikit-learn) avec des variables de graphe calculées à la main ;
  - GNN : réseau de neurones de graphe spatio-temporel léger (gnn.py).
Et deux références : profil moyen (station x jour x heure) et HGB sans aucune variable de trafic,
pour mesurer ce que le trafic apporte vraiment.

Chaque modèle est évalué deux fois sur la période de test :
  - "trafic observé" : comme si l'étape 1 était parfaite (borne haute) ;
  - "trafic prévu"   : chaîne complète, l'étape 2 reçoit la prévision de l'étape 1 (ce qu'on aura en vrai).

Source du trafic (--trafic) :
  - here         : tronçons HERE seuls (here/here_troncons.csv, format de here_flux.py) ;
  - here+avatar  : tronçons HERE et compteurs Avatar, pour mesurer ce qu'Avatar ajoute ;
  - avatar       : compteurs Avatar seuls ;
  - auto (défaut): here si le fichier HERE existe, sinon avatar.

Usage :
    python entrainer.py --donnees donnees_synthetiques --horizon-h 24 --sortie resultats
    python entrainer.py --donnees donnees_reelles --trafic here+avatar \
        --compteurs donnees_reelles/rattachement_interurbains/compteurs_retenus.csv --sortie resultats_here_avatar
"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

import graphe
from donnees import VARIABLES_VOLUME, charger
from gnn import entrainer_gnn, predire_gnn

SEUILS_COULEUR = [0.40, 0.70, 0.90]  # vert < 40 % <= orange < 70 % <= rouge < 90 % <= noir
COULEURS = np.array(["vert", "orange", "rouge", "noir"])


def couleur(taux):
    return np.digitize(taux, SEUILS_COULEUR)


def variables_calendaires(cal: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Heure et jour en cycles, puis toutes les variables numériques du calendrier : celles de
    calendrier.py (vacances par zone et rang dans la période, fériés, ponts, Bison Futé par sens) ou
    celles du jeu synthétique (vacances, début/fin, bison_fute_couleur)."""
    h, j = cal["heure"].values, cal["jour_semaine"].values
    cols = {
        "heure_sin": np.sin(2 * np.pi * h / 24), "heure_cos": np.cos(2 * np.pi * h / 24),
        "jour_sin": np.sin(2 * np.pi * j / 7), "jour_cos": np.cos(2 * np.pi * j / 7),
        "heure": h, "jour_semaine": j, "mois": cal["mois"].values,
    }
    for c in cal.select_dtypes("number").columns:
        if c not in cols:
            cols[c] = cal[c].values
    return np.column_stack(list(cols.values())).astype(np.float32), list(cols)


def decalages_autorises(horizon_pas, pas_par_jour):
    """Retards utilisables quand on prévoit à H pas : uniquement ceux >= H."""
    cands = [horizon_pas, pas_par_jour, 2 * pas_par_jour, 7 * pas_par_jour, 14 * pas_par_jour]
    return sorted({k for k in cands if k >= horizon_pas})


def moyenne_glissante_decalee(x, fenetre, decalage):
    """Moyenne de x sur [t - decalage - fenetre + 1, t - decalage], colonne par colonne."""
    df = pd.DataFrame(x).rolling(fenetre, min_periods=1).mean().values
    return graphe.decaler(df, decalage)


def metriques(y, p):
    m = ~np.isnan(y)
    y, p = y[m], p[m]
    cy, cp = couleur(y), couleur(p)
    alerte = cy >= 2
    return {
        "MAE_points": round(float(np.mean(np.abs(y - p)) * 100), 2),
        "RMSE_points": round(float(np.sqrt(np.mean((y - p) ** 2))) * 100, 2),
        "couleur_exacte": round(float(np.mean(cy == cp)), 3),
        "rappel_alertes_rouge_noir": round(float(np.mean(cp[alerte] >= 2)), 3) if alerte.any() else None,
        "precision_alertes": round(float(np.mean(cy[cp >= 2] >= 2)), 3) if (cp >= 2).any() else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donnees", default="donnees_synthetiques")
    ap.add_argument("--sortie", default="resultats")
    ap.add_argument("--trafic", default="auto", choices=["auto", "here", "here+avatar", "avatar"])
    ap.add_argument("--here", help="here_troncons.csv (défaut : <donnees>/here/here_troncons.csv)")
    ap.add_argument("--compteurs", help="liste des compteurs Avatar à garder, ex. rattachement_interurbains/compteurs_retenus.csv")
    ap.add_argument("--pas", default="1h")
    ap.add_argument("--horizon-h", type=float, default=24, help="horizon de prévision en heures")
    ap.add_argument("--fenetre-gnn-h", type=float, default=6, help="historique vu par le GNN")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--sans-gnn", action="store_true")
    ap.add_argument("--part-train", type=float, default=0.70)
    ap.add_argument("--part-val", type=float, default=0.15)
    ap.add_argument("--lignes-max-etape1", type=int, default=3_000_000,
                    help="lignes tirées au hasard pour apprendre l'étape 1 (mémoire)")
    args = ap.parse_args()
    out = Path(args.sortie)
    out.mkdir(parents=True, exist_ok=True)

    here = Path(args.here) if args.here else Path(args.donnees) / "here" / "here_troncons.csv"
    trafic = args.trafic if args.trafic != "auto" else ("here" if here.exists() else "avatar")
    print(f"Trafic : {trafic}")
    d = charger(args.donnees, args.pas, trafic=trafic, here=here, fichier_compteurs=args.compteurs)
    T, S, C = len(d.index), len(d.stations), len(d.compteurs)
    V = len(d.variables_trafic)
    pas_min = pd.Timedelta(args.pas) / pd.Timedelta(minutes=1)
    pas_par_jour = int(round(24 * 60 / pas_min))
    H = int(round(args.horizon_h * 60 / pas_min))
    lags = decalages_autorises(H, pas_par_jour)
    print(f"Horizon {args.horizon_h} h = {H} pas ; retards d'occupation utilisables : {lags} pas")

    # ---------------- Découpage temporel (jamais aléatoire : fuite d'information sinon)
    n_tr, n_va = int(T * args.part_train), int(T * args.part_val)
    i_tr, i_va, i_te = np.arange(T)[:n_tr], np.arange(T)[n_tr:n_tr + n_va], np.arange(T)[n_tr + n_va:]
    for nom, ii in (("apprentissage", i_tr), ("validation", i_va), ("test", i_te)):
        print(f"  {nom:14s} {d.index[ii[0]]:%Y-%m-%d} -> {d.index[ii[-1]]:%Y-%m-%d}")

    # ---------------- Graphe
    noeuds = graphe.table_noeuds(d.stations, d.compteurs)
    # un aretes.csv fourni ne connaît que les compteurs Avatar : avec HERE, le graphe est reconstruit
    aretes = d.aretes if d.aretes is not None and trafic == "avatar" else graphe.construire_aretes(noeuds)
    aretes.to_csv(out / "aretes_utilisees.csv", index=False)
    temps = graphe.matrice_temps(noeuds, aretes)
    temps_cs = temps[S:, :S]  # noeud de trafic -> station
    temps_sc = temps[:S, S:]  # station -> noeud de trafic en aval
    A_amont, A_aval = graphe.adjacences(temps)
    print(f"Graphe : {len(noeuds)} noeuds, {len(aretes)} arêtes ; "
          f"{np.isfinite(temps_cs).any(0).sum()}/{S} stations ont au moins un noeud de trafic en amont")

    # ---------------- Normalisation : les volumes en multiple de la moyenne du noeud sur l'apprentissage
    trafic_n = d.trafic.copy()
    normalisation = {}
    for k, v in enumerate(d.variables_trafic):
        if v in VARIABLES_VOLUME:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # noeuds d'une autre source : que des NaN
                moy = np.nanmean(d.trafic[i_tr, :, k], axis=0)
            trafic_n[..., k] = d.trafic[..., k] / np.where(moy > 0, moy, np.nan)
            normalisation[v] = {n: float(m) for n, m in zip(d.compteurs["compteur_id"], moy) if np.isfinite(m)}
    Xcal, noms_cal = variables_calendaires(d.calendrier)

    # ================= ÉTAPE 1 : prévision du trafic aux noeuds de trafic =================
    print("\nÉtape 1 : prévision du trafic")
    axe_code = pd.factorize(d.compteurs["axe_rattache"])[0]
    stat1 = np.column_stack([np.arange(C), d.compteurs["pk_projete"].values, d.compteurs["sens"].values, axe_code]).astype(np.float32)
    noms_stat1 = ["noeud", "pk", "sens", "axe"]
    rng = np.random.default_rng(0)
    trafic_prevu_n = np.full_like(trafic_n, np.nan)
    etape1, err1 = {}, {}
    for k, v in enumerate(d.variables_trafic):
        x = trafic_n[..., k]
        cols = np.where(~np.isnan(x[i_tr]).all(0))[0]  # noeuds qui portent cette variable
        x = x[:, cols]
        lags_v = [graphe.decaler(x, L) for L in lags]
        moy24 = moyenne_glissante_decalee(x, pas_par_jour, H)
        noms1 = noms_cal + noms_stat1 + [f"{v}_lag{L}" for L in lags] + [f"{v}_moy24h_decale"]

        def table1(ii):
            n = len(ii)
            blocs = [np.repeat(Xcal[ii, None, :], len(cols), 1), np.repeat(stat1[None, cols], n, 0)]
            blocs += [l[ii, :, None] for l in lags_v] + [moy24[ii, :, None]]
            return np.concatenate(blocs, -1).astype(np.float32)

        # apprentissage sur un tirage de pas de temps, pour borner la mémoire
        n_pas = min(len(i_tr), max(1, args.lignes_max_etape1 // len(cols)))
        ii = np.sort(rng.choice(i_tr, n_pas, replace=False))
        X1, y1 = table1(ii).reshape(-1, len(noms1)), x[ii].ravel()
        ok = ~np.isnan(y1)
        cat = [noms1.index("noeud")] if len(cols) <= 255 else None
        mdl = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, categorical_features=cat,
                                            early_stopping=True, random_state=0)
        mdl.fit(X1[ok], y1[ok])
        prev = np.concatenate([mdl.predict(table1(np.arange(T)[a:a + 256]).reshape(-1, len(noms1))).reshape(-1, len(cols))
                               for a in range(0, T, 256)])
        trafic_prevu_n[:, cols, k] = prev
        yt, pt = x[i_te].ravel(), prev[i_te].ravel()
        m = ~np.isnan(yt)
        if v in VARIABLES_VOLUME:  # erreur relative au trafic moyen
            err1[v] = float(np.mean(np.abs(yt[m] - pt[m])) / np.mean(yt[m]))
            print(f"  {v:12s} {len(cols):5d} noeuds, erreur relative moyenne sur le test : {err1[v]:.1%}")
        else:  # variable déjà entre 0 et 1 : erreur absolue
            err1[v] = float(np.mean(np.abs(yt[m] - pt[m])))
            print(f"  {v:12s} {len(cols):5d} noeuds, erreur absolue moyenne sur le test : {err1[v]:.3f}")
        etape1[v] = {"modele": mdl, "noeuds": d.compteurs["compteur_id"].values[cols].tolist(), "noms": noms1}
    # pour l'étape 2, les trous de mesure observés sont comblés par la prévision
    trafic_obs_n = np.where(np.isnan(trafic_n), trafic_prevu_n, trafic_n)

    # ================= ÉTAPE 2 : occupation des stations =================
    occ = d.occ
    occ_lags = {k: graphe.decaler(occ, k) for k in lags}
    occ_moy_jour = moyenne_glissante_decalee(occ, pas_par_jour, H)
    # occupation des stations voisines (amont/aval), connue il y a H pas
    A_ss_am, A_ss_av = A_amont[:S, :S], A_aval[:S, :S]

    def voisinage(A, x):
        v = np.nan_to_num(x) @ A.T
        poids = (~np.isnan(x)).astype(float) @ A.T
        return np.where(poids > 0, v / np.maximum(poids, 1e-9), np.nan)

    voisins_amont, voisins_aval = voisinage(A_ss_am, occ_lags[H]), voisinage(A_ss_av, occ_lags[H])

    st = d.stations
    statiques = np.column_stack([np.arange(S), st["nb_pdc"].values, st["pk"].values,
                                 st.get("puissance_kw", pd.Series(0, index=st.index)).values, st["sens"].values])
    noms_stat = ["station", "nb_pdc", "pk", "puissance_kw", "sens"]

    def variables_trafic_stations(tr_n):
        """Pour chaque variable de trafic : trafic amont propagé (avec le retard de parcours) et valeur du
        noeud le plus proche en aval. Renvoie un dict nom -> [T, S]."""
        res = {}
        for k, v in enumerate(d.variables_trafic):
            cols = np.where(~np.isnan(tr_n[..., k]).all(0))[0]  # noeuds qui portent cette variable
            for nom, x in graphe.variables_propagation(tr_n[:, cols, k], temps_cs[cols], pas_min).items():
                res[f"{v}_{nom}"] = x
            res[f"{v}_aval_proche"] = graphe.variables_aval(tr_n[:, cols, k], temps_sc[:, cols])
        return res

    noms_trafic_vides = {n for n, v in variables_trafic_stations(trafic_obs_n).items() if np.isnan(v[i_tr]).all()}

    def table_hgb(tr_n, avec_trafic=True):
        """Une ligne par (pas de temps, station). Renvoie [T, S, F] et noms."""
        blocs, noms = [np.repeat(Xcal[:, None, :], S, 1), np.repeat(statiques[None], T, 0)], noms_cal + noms_stat
        for k in lags:
            blocs.append(occ_lags[k][..., None]); noms.append(f"occ_lag{k}")
        blocs += [occ_moy_jour[..., None], voisins_amont[..., None], voisins_aval[..., None]]
        noms += ["occ_moy24h_decale", "occ_voisins_amont_lagH", "occ_voisins_aval_lagH"]
        if avec_trafic:
            for nom, v in variables_trafic_stations(tr_n).items():
                if nom in noms_trafic_vides:  # ex. aucun compteur Avatar n'est en aval d'une station
                    continue
                blocs.append(v[..., None]); noms.append(nom)
        return np.concatenate(blocs, -1).astype(np.float32), noms
    def fit_hgb(Xs, noms):
        y = occ[i_tr].ravel()
        Xf = Xs[i_tr].reshape(-1, Xs.shape[-1])
        ok = ~np.isnan(y)
        mdl = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                            # identifiant station en catégoriel (limité à 255 modalités)
                                            categorical_features=[noms.index("station")] if S <= 255 else None,
                                            early_stopping=True,
                                            random_state=0)
        return mdl.fit(Xf[ok], y[ok])

    def pred_hgb(mdl, Xs, ii):
        return np.clip(mdl.predict(Xs[ii].reshape(-1, Xs.shape[-1])).reshape(len(ii), S), 0, 1)

    resultats, predictions = {}, {}
    y_te = occ[i_te]

    # Référence 1 : profil moyen station x jour x heure
    cal = d.calendrier
    cle = cal["jour_semaine"].values * 1000 + (cal["heure"].values * 10).astype(int)
    prof = pd.DataFrame(occ[i_tr]).groupby(cle[i_tr]).mean()
    p_prof = prof.reindex(cle[i_te]).values
    resultats["profil_moyen"] = metriques(y_te, p_prof)
    predictions["profil_moyen"] = p_prof

    print("\nÉtape 2 : gradient boosting")
    X_sans, noms_sans = table_hgb(trafic_obs_n, avec_trafic=False)
    hgb_sans = fit_hgb(X_sans, noms_sans)
    predictions["hgb_sans_trafic"] = pred_hgb(hgb_sans, X_sans, i_te)
    resultats["hgb_sans_trafic"] = metriques(y_te, predictions["hgb_sans_trafic"])

    X_obs, noms2 = table_hgb(trafic_obs_n)
    X_prev, _ = table_hgb(trafic_prevu_n)
    hgb = fit_hgb(X_obs, noms2)
    predictions["hgb_trafic_observe"] = pred_hgb(hgb, X_obs, i_te)
    predictions["hgb_trafic_prevu"] = pred_hgb(hgb, X_prev, i_te)
    resultats["hgb_trafic_observe"] = metriques(y_te, predictions["hgb_trafic_observe"])
    resultats["hgb_trafic_prevu"] = metriques(y_te, predictions["hgb_trafic_prevu"])
    joblib.dump({"etape1": etape1, "etape2": hgb, "noms_etape2": noms2}, out / "modeles_hgb.joblib")

    # ---------------- GNN
    if not args.sans_gnn:
        print("\nÉtape 2 : GNN spatio-temporel")
        T_in = max(1, int(round(args.fenetre_gnn_h * 60 / pas_min)))

        types = d.compteurs.get("source", pd.Series("avatar", index=d.compteurs.index)).values

        def tenseur_gnn(tr_n):
            """Variables propres à chaque noeud : trafic (0 pour une station ou une variable absente),
            type de noeud, occupation passée, capacité. Le calendrier est passé à part."""
            N = S + C
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

        # calendrier commun, ramené à une échelle de l'ordre de 1 (rang dans les vacances, etc.)
        cal_sel = Xcal[:, [noms_cal.index(n) for n in noms_cal if n not in ("heure", "jour_semaine", "mois")]]
        cal_g = (cal_sel / np.maximum(1, np.abs(cal_sel[i_tr]).max(0))).astype(np.float32)
        G_obs, G_prev = tenseur_gnn(trafic_obs_n), tenseur_gnn(trafic_prevu_n)
        debut = max(T_in - 1, max(lags))  # pas de fenêtre incomplète
        tr, va = i_tr[i_tr >= debut], i_va
        modele = entrainer_gnn(G_obs, occ, tr, va, A_amont, A_aval, T_in, S, cal=cal_g, epochs=args.epochs)
        predictions["gnn_trafic_observe"] = predire_gnn(modele, G_obs, i_te, T_in, S, cal=cal_g)
        predictions["gnn_trafic_prevu"] = predire_gnn(modele, G_prev, i_te, T_in, S, cal=cal_g)
        resultats["gnn_trafic_observe"] = metriques(y_te, predictions["gnn_trafic_observe"])
        resultats["gnn_trafic_prevu"] = metriques(y_te, predictions["gnn_trafic_prevu"])
        torch.save({"etat": modele.state_dict(), "n_feat": G_obs.shape[-1], "n_cal": cal_g.shape[1], "T_in": T_in,
                    "adaptative": modele.adaptative}, out / "modele_gnn.pt")

    # ---------------- Restitution
    tab = pd.DataFrame(resultats).T
    print("\nRésultats sur la période de test (MAE et RMSE en points de taux d'occupation) :")
    print(tab.to_string())
    tab.to_csv(out / "metriques.csv")
    with open(out / "config.json", "w") as f:
        json.dump({"pas": args.pas, "horizon_h": args.horizon_h, "horizon_pas": H, "lags": lags, "trafic": trafic,
                   "variables_trafic": d.variables_trafic, "seuils_couleur": SEUILS_COULEUR,
                   "moyennes_normalisation": normalisation, "stations": d.stations["station_id"].tolist(),
                   "noeuds_trafic": d.compteurs["compteur_id"].tolist(), "erreur_etape1": err1}, f, indent=2, ensure_ascii=False)

    lignes = []
    for nom, p in predictions.items():
        df = pd.DataFrame(p, index=d.index[i_te], columns=d.stations["station_id"]).stack().rename("prevu")
        lignes.append(df.to_frame().assign(modele=nom))
    pr = pd.concat(lignes).reset_index().rename(columns={"level_0": "horodatage"})
    reel = pd.DataFrame(y_te, index=d.index[i_te], columns=d.stations["station_id"]).stack().rename("reel").reset_index()
    reel.columns = ["horodatage", "station_id", "reel"]
    pr.columns = ["horodatage", "station_id", "prevu", "modele"]
    pr = pr.merge(reel, on=["horodatage", "station_id"], how="left")
    pr["couleur_prevue"] = COULEURS[couleur(pr["prevu"].values)]
    pr.to_csv(out / "predictions_test.csv", index=False)
    print(f"\nModèles, métriques et prévisions écrits dans {out}/")


if __name__ == "__main__":
    main()
