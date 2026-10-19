"""
Rattache les compteurs Avatar au graphe des stations d'autoroute et mesure, station par station,
si le trafic amont propagé est lié à l'occupation.

Entrées (dossier --donnees, par défaut donnees_reelles/) :
  stations.csv, occupation_bornes.csv, debits/compteurs.csv, debits/comptages.csv

Sorties (dossier --sortie, par défaut donnees_reelles/rattachement/) :
  aretes.csv                 graphe utilisé (stations + compteurs retenus)
  compteurs_retenus.csv      compteurs gardés après filtrage qualité
  stations_compteurs.csv     par station : compteurs amont, plus proche, temps, corrélations
  liens_compteur_station.csv par couple compteur -> station en amont : temps, retard, poids, corrélations

Deux corrélations, calculées sur les heures où l'occupation est connue :
  - brute : trafic amont propagé x occupation. Elle est gonflée par le cycle jour / nuit commun aux deux.
  - anomalie : même chose après avoir retiré à chaque série sa moyenne par heure de la journée.
    C'est elle qui dit si le compteur apporte une information au-delà de l'heure.

Usage :
    python rattacher_compteurs.py --donnees donnees_reelles --part-estimee-max 50
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import graphe
from donnees import _trouver, lire_comptages


def correlation(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 24 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def anomalie(x: np.ndarray, heures: np.ndarray) -> np.ndarray:
    """Retire à chaque colonne sa moyenne par heure de la journée."""
    df = pd.DataFrame(x)
    return (df - df.groupby(heures).transform("mean")).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donnees", default="donnees_reelles")
    ap.add_argument("--sortie", default=None)
    ap.add_argument("--avatar", help="dossier compteurs.csv + comptages (défaut : <donnees>/debits), ex. data_IRVE/avatar")
    ap.add_argument("--occupation", help="occupation_bornes.csv[.xz] s'il n'est pas dans --donnees")
    ap.add_argument("--part-estimee-max", type=float, default=50.0,
                    help="heure de comptage écartée si plus de ce % est estimé par Avatar")
    ap.add_argument("--couverture-min", type=float, default=0.5,
                    help="compteur écarté s'il a moins de cette part d'heures mesurées sur la période d'occupation")
    ap.add_argument("--interurbains-seulement", action="store_true",
                    help="ne garder que les compteurs à plus de 30 km des extrémités urbaines des axes")
    ap.add_argument("--traces", help="axes_osm.csv : recalcule le pk des compteurs sur ce tracé")
    ap.add_argument("--tmax-min", type=float, default=300.0)
    ap.add_argument("--tau0-min", type=float, default=90.0)
    args = ap.parse_args()
    d = Path(args.donnees)
    out = Path(args.sortie) if args.sortie else d / "rattachement"
    out.mkdir(parents=True, exist_ok=True)

    av = Path(args.avatar) if args.avatar else d / "debits"
    st = pd.read_csv(_trouver(d, "stations.csv"))
    occ = pd.read_csv(args.occupation or _trouver(d, "occupation_bornes.csv"), parse_dates=["horodatage"])
    occ["taux"] = (occ["nb_pdc_occupes"] / occ["nb_pdc_statut_connu"].where(occ["nb_pdc_statut_connu"] > 0)).clip(0, 1)
    cpt = pd.read_csv(_trouver(av, "compteurs.csv"))
    if args.traces:  # pk des compteurs recalculés sur le même tracé que les stations
        from reperer_stations_autoroute import projeter_sur_axe
        tr = {a: t.sort_values("pk").reset_index(drop=True) for a, t in pd.read_csv(args.traces).groupby("axe")}
        cpt["pk_projete"] = [round(projeter_sur_axe(r.lat, r.lon, tr[r.axe_rattache])[1], 2)
                             if r.axe_rattache in tr else r.pk_projete for r in cpt.itertuples()]
    q = lire_comptages(av, parse_dates=["horodatage"])

    # -------- qualité des comptages : on écarte les heures surtout estimées par Avatar
    q.loc[q["part_estimee_pct"] > args.part_estimee_max, "debit_vh"] = np.nan
    M = q.pivot_table(index="horodatage", columns="compteur_id", values="debit_vh")
    index = pd.date_range(M.index.min(), M.index.max(), freq="1h")
    M = M.reindex(index)

    # occupation sur la même grille (seules les heures du 9 au 15 juillet sont renseignées)
    Y = occ.pivot_table(index="horodatage", columns="station_id", values="taux").reindex(index)
    st = st[st["station_id"].isin(Y.columns)].reset_index(drop=True)
    Y = Y[st["station_id"]]
    periode = Y.notna().any(axis=1).values

    couverture = M[periode].notna().mean()
    garde = couverture[couverture >= args.couverture_min].index
    cpt = cpt[cpt["compteur_id"].isin(garde)]
    if args.interurbains_seulement:
        cpt = cpt[cpt["hors_agglo_extremites"].astype(bool)]
    cpt = cpt.reset_index(drop=True)
    M = M[cpt["compteur_id"]]
    print(f"{len(cpt)} compteurs retenus sur {len(couverture)} (heures estimées > {args.part_estimee_max:.0f} % écartées, "
          f"couverture >= {args.couverture_min:.0%} du 9 au 15 juillet)")
    cpt.to_csv(out / "compteurs_retenus.csv", index=False)

    # -------- graphe
    noeuds = graphe.table_noeuds(st, cpt)
    aretes = graphe.construire_aretes(noeuds)
    aretes.to_csv(out / "aretes.csv", index=False)
    T = graphe.matrice_temps(noeuds, aretes)
    S = len(st)
    temps_cs = T[S:, :S]
    temps_cs[temps_cs > args.tmax_min] = np.inf
    print(f"Graphe : {len(noeuds)} noeuds, {len(aretes)} arêtes")

    # -------- variables de propagation et corrélations
    qn = (M / M.mean()).values  # 1 = trafic moyen du compteur sur mai-juillet
    prop = graphe.variables_propagation(qn, temps_cs, 60.0, tau0_min=args.tau0_min, tmax_min=args.tmax_min)
    x0 = prop["trafic_amont_propage_t-0"]
    heures = index.hour.values
    y = Y.values
    ya, xa = anomalie(np.where(periode[:, None], y, np.nan), heures), anomalie(np.where(periode[:, None], x0, np.nan), heures)

    lignes_st, lignes_lien = [], []
    for s in range(S):
        amont = np.where(np.isfinite(temps_cs[:, s]))[0]
        r_brut, r_anom = correlation(x0[:, s], y[:, s]), correlation(xa[:, s], ya[:, s])
        proche = amont[np.argmin(temps_cs[amont, s])] if len(amont) else None
        lignes_st.append(dict(
            station_id=st.at[s, "station_id"], nom=st.at[s, "nom"], axe=st.at[s, "axe"], sens=st.at[s, "sens"],
            pk=st.at[s, "pk"], occupation_moyenne=round(float(np.nanmean(y[:, s])), 3),
            nb_compteurs_amont=len(amont),
            compteur_proche=cpt.at[proche, "compteur_id"] if proche is not None else None,
            route_proche=cpt.at[proche, "nom_station"] if proche is not None else None,
            temps_proche_min=round(float(temps_cs[proche, s]), 1) if proche is not None else None,
            r_brut=round(r_brut, 3) if np.isfinite(r_brut) else None,
            r_anomalie=round(r_anom, 3) if np.isfinite(r_anom) else None))
        for c in amont:
            retard = int(round(temps_cs[c, s] / 60.0))
            xc = graphe.decaler(qn[:, c], retard)
            xca = anomalie(np.where(periode, xc, np.nan)[:, None], heures)[:, 0]
            lignes_lien.append(dict(
                compteur_id=cpt.at[c, "compteur_id"], station_id=st.at[s, "station_id"],
                temps_min=round(float(temps_cs[c, s]), 1), retard_h=retard,
                poids=round(float(np.exp(-temps_cs[c, s] / args.tau0_min)), 3),
                r_brut=correlation(xc, y[:, s]), r_anomalie=correlation(xca, ya[:, s])))

    ts = pd.DataFrame(lignes_st).sort_values(["axe", "sens", "pk"])
    ts.to_csv(out / "stations_compteurs.csv", index=False)
    tl = pd.DataFrame(lignes_lien).round(3)
    tl.to_csv(out / "liens_compteur_station.csv", index=False)

    avec = ts[ts["nb_compteurs_amont"] > 0]
    print(f"\n{len(avec)}/{S} stations ont au moins un compteur en amont à moins de {args.tmax_min:.0f} min")
    print(f"Corrélation trafic amont / occupation, médiane : brute {avec['r_brut'].median():.2f}, "
          f"hors cycle journalier {avec['r_anomalie'].median():.2f}")
    print(avec.groupby("axe").agg(stations=("station_id", "size"), compteurs_moy=("nb_compteurs_amont", "mean"),
                                  temps_proche_med=("temps_proche_min", "median"), r_brut=("r_brut", "median"),
                                  r_anomalie=("r_anomalie", "median")).round(2).to_string())
    print(f"\nFichiers écrits dans {out}/")


if __name__ == "__main__":
    main()
