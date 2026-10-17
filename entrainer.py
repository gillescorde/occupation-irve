"""
Script d'apprentissage du "Bison Futé des bornes".

Chaîne en deux étapes (voir README.md) :
  Étape 1 : prévoir le débit aux compteurs routiers à l'horizon H (calendrier + historique du compteur).
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

Usage :
    python entrainer.py --donnees donnees_synthetiques --horizon-h 24 --sortie resultats
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

import graphe
from donnees import charger
from gnn import entrainer_gnn, predire_gnn

SEUILS_COULEUR = [0.40, 0.70, 0.90]  # vert < 40 % <= orange < 70 % <= rouge < 90 % <= noir
COULEURS = np.array(["vert", "orange", "rouge", "noir"])


def couleur(taux):
    return np.digitize(taux, SEUILS_COULEUR)


def variables_calendaires(cal: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    h, j = cal["heure"].values, cal["jour_semaine"].values
    cols = {
        "heure_sin": np.sin(2 * np.pi * h / 24), "heure_cos": np.cos(2 * np.pi * h / 24),
        "jour_sin": np.sin(2 * np.pi * j / 7), "jour_cos": np.cos(2 * np.pi * j / 7),
        "heure": h, "jour_semaine": j, "mois": cal["mois"].values,
        "ferie": cal["ferie"].values, "vacances": cal["vacances"].values,
        "debut_vacances": cal["debut_vacances"].values, "fin_vacances": cal["fin_vacances"].values,
        "bison_fute_couleur": cal["bison_fute_couleur"].values,
    }
    return np.column_stack(list(cols.values())).astype(float), list(cols)


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
    ap.add_argument("--pas", default="1h")
    ap.add_argument("--horizon-h", type=float, default=24, help="horizon de prévision en heures")
    ap.add_argument("--fenetre-gnn-h", type=float, default=6, help="historique vu par le GNN")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--sans-gnn", action="store_true")
    ap.add_argument("--part-train", type=float, default=0.70)
    ap.add_argument("--part-val", type=float, default=0.15)
    args = ap.parse_args()
    out = Path(args.sortie)
    out.mkdir(parents=True, exist_ok=True)

    d = charger(args.donnees, args.pas)
    T, S, C = len(d.index), len(d.stations), len(d.compteurs)
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
    aretes = d.aretes if d.aretes is not None else graphe.construire_aretes(noeuds)
    aretes.to_csv(out / "aretes_utilisees.csv", index=False)
    temps = graphe.matrice_temps(noeuds, aretes)
    temps_cs = temps[S:, :S]  # compteur -> station
    A_amont, A_aval = graphe.adjacences(temps)
    print(f"Graphe : {len(noeuds)} noeuds, {len(aretes)} arêtes ; "
          f"{np.isfinite(temps_cs).any(0).sum()}/{S} stations ont au moins un compteur en amont")

    # ---------------- Normalisation des débits (1 = moyenne du compteur sur l'apprentissage)
    moy_cpt = np.nanmean(d.debit[i_tr], axis=0)
    debit_n = d.debit / moy_cpt
    Xcal, noms_cal = variables_calendaires(d.calendrier)

    # ================= ÉTAPE 1 : prévision du trafic aux compteurs =================
    print("\nÉtape 1 : prévision du débit aux compteurs")
    lags_cpt = [graphe.decaler(debit_n, k) for k in lags]
    moy24 = moyenne_glissante_decalee(debit_n, pas_par_jour, H)
    lignes = []
    for c in range(C):
        lignes.append(np.column_stack([Xcal, np.full(T, c)] + [l[:, c] for l in lags_cpt] + [moy24[:, c]]))
    X1 = np.stack(lignes, 1)  # [T, C, F1]
    noms1 = noms_cal + ["compteur"] + [f"debit_lag{k}" for k in lags] + ["debit_moy24h_decale"]
    ok = ~np.isnan(debit_n[i_tr]).ravel()
    etape1 = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, categorical_features=[noms1.index("compteur")],
                                           early_stopping=True, random_state=0)
    etape1.fit(X1[i_tr].reshape(-1, X1.shape[-1])[ok], debit_n[i_tr].ravel()[ok])
    debit_prevu_n = etape1.predict(X1.reshape(-1, X1.shape[-1])).reshape(T, C)
    yt, pt = debit_n[i_te].ravel(), debit_prevu_n[i_te].ravel()
    m = ~np.isnan(yt)
    err1 = float(np.mean(np.abs(yt[m] - pt[m]) / np.mean(yt[m])))
    print(f"  Erreur relative moyenne sur le test : {err1:.1%}")
    # pour l'étape 2, les trous de mesure observés sont comblés par la prévision
    debit_obs_n = np.where(np.isnan(debit_n), debit_prevu_n, debit_n)

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

    def table_hgb(debit_norm, avec_trafic=True):
        """Une ligne par (pas de temps, station). Renvoie [T, S, F] et noms."""
        blocs, noms = [np.repeat(Xcal[:, None, :], S, 1), np.repeat(statiques[None], T, 0)], noms_cal + noms_stat
        for k in lags:
            blocs.append(occ_lags[k][..., None]); noms.append(f"occ_lag{k}")
        blocs += [occ_moy_jour[..., None], voisins_amont[..., None], voisins_aval[..., None]]
        noms += ["occ_moy24h_decale", "occ_voisins_amont_lagH", "occ_voisins_aval_lagH"]
        if avec_trafic:
            prop = graphe.variables_propagation(debit_norm, temps_cs, pas_min)
            for nom, v in prop.items():
                blocs.append(v[..., None]); noms.append(nom)
        return np.concatenate(blocs, -1), noms

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
    X_sans, noms_sans = table_hgb(debit_obs_n, avec_trafic=False)
    hgb_sans = fit_hgb(X_sans, noms_sans)
    predictions["hgb_sans_trafic"] = pred_hgb(hgb_sans, X_sans, i_te)
    resultats["hgb_sans_trafic"] = metriques(y_te, predictions["hgb_sans_trafic"])

    X_obs, noms2 = table_hgb(debit_obs_n)
    X_prev, _ = table_hgb(debit_prevu_n)
    hgb = fit_hgb(X_obs, noms2)
    predictions["hgb_trafic_observe"] = pred_hgb(hgb, X_obs, i_te)
    predictions["hgb_trafic_prevu"] = pred_hgb(hgb, X_prev, i_te)
    resultats["hgb_trafic_observe"] = metriques(y_te, predictions["hgb_trafic_observe"])
    resultats["hgb_trafic_prevu"] = metriques(y_te, predictions["hgb_trafic_prevu"])
    joblib.dump({"etape1": etape1, "etape2": hgb, "noms_etape1": noms1, "noms_etape2": noms2}, out / "modeles_hgb.joblib")

    # ---------------- GNN
    if not args.sans_gnn:
        print("\nÉtape 2 : GNN spatio-temporel")
        T_in = max(1, int(round(args.fenetre_gnn_h * 60 / pas_min)))

        def tenseur_gnn(debit_norm):
            N = S + C
            f_debit = np.zeros((T, N)); f_debit[:, S:] = np.nan_to_num(debit_norm)
            f_type = np.zeros((T, N)); f_type[:, S:] = 1
            f_occ = np.zeros((T, N, len(lags) + 1)); m_occ = np.zeros((T, N))
            for j, k in enumerate(lags):
                f_occ[:, :S, j] = np.nan_to_num(occ_lags[k], nan=0)
            f_occ[:, :S, -1] = np.nan_to_num(occ_moy_jour)
            m_occ[:, :S] = ~np.isnan(occ_lags[H])
            cal_sel = Xcal[:, [noms_cal.index(n) for n in noms_cal if n not in ("heure", "jour_semaine", "mois")]]
            cal_b = np.repeat(cal_sel[:, None, :], N, 1)
            cap = np.zeros((T, N)); cap[:, :S] = st["nb_pdc"].values / 10
            return np.concatenate([f_debit[..., None], f_type[..., None], f_occ, m_occ[..., None], cal_b,
                                   cap[..., None]], -1).astype(np.float32)

        G_obs, G_prev = tenseur_gnn(debit_obs_n), tenseur_gnn(debit_prevu_n)
        debut = max(T_in - 1, max(lags))  # pas de fenêtre incomplète
        tr, va = i_tr[i_tr >= debut], i_va
        modele = entrainer_gnn(G_obs, occ, tr, va, A_amont, A_aval, T_in, S, epochs=args.epochs)
        predictions["gnn_trafic_observe"] = predire_gnn(modele, G_obs, i_te, T_in, S)
        predictions["gnn_trafic_prevu"] = predire_gnn(modele, G_prev, i_te, T_in, S)
        resultats["gnn_trafic_observe"] = metriques(y_te, predictions["gnn_trafic_observe"])
        resultats["gnn_trafic_prevu"] = metriques(y_te, predictions["gnn_trafic_prevu"])
        torch.save({"etat": modele.state_dict(), "n_feat": G_obs.shape[-1], "T_in": T_in}, out / "modele_gnn.pt")

    # ---------------- Restitution
    tab = pd.DataFrame(resultats).T
    print("\nRésultats sur la période de test (MAE et RMSE en points de taux d'occupation) :")
    print(tab.to_string())
    tab.to_csv(out / "metriques.csv")
    with open(out / "config.json", "w") as f:
        json.dump({"pas": args.pas, "horizon_h": args.horizon_h, "horizon_pas": H, "lags": lags,
                   "seuils_couleur": SEUILS_COULEUR, "moyenne_compteurs": dict(zip(d.compteurs["compteur_id"], moy_cpt.tolist())),
                   "stations": d.stations["station_id"].tolist(), "compteurs": d.compteurs["compteur_id"].tolist(),
                   "erreur_etape1_relative": err1}, f, indent=2, ensure_ascii=False)

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
