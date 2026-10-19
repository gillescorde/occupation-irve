"""
Représentation du réseau routier sous forme de graphe orienté, et variables de propagation.

Noeuds : les stations de recharge ET les noeuds de trafic (compteurs Avatar, tronçons HERE).
Arêtes : orientées dans le sens de circulation, pondérées par le temps de parcours (minutes).
  - le long d'un axe et d'un sens : chaîne des stations et des tronçons HERE, chacun vers le suivant
    (un tronçon HERE est SUR l'autoroute : un bouchon en aval d'une aire se voit donc aussi) ;
  - compteur Avatar (sur une nationale parallèle ou une entrée) -> premier noeud de la chaîne en aval.

À partir de là on dérive deux choses :
  1. une matrice des temps de parcours entre tout couple de noeuds (plus court chemin) ;
  2. pour le gradient boosting : des variables "trafic amont propagé", c'est-à-dire le débit
     des compteurs en amont, décalé du temps qu'il faut aux véhicules pour atteindre la station ;
  3. pour le GNN : des matrices d'adjacence (amont et aval) que le réseau de neurones utilise
     pour faire circuler l'information entre noeuds.

Convention : sens 1 = sens des pk croissants, sens 2 = pk décroissants, 0 = inconnu ou les deux.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def table_noeuds(stations: pd.DataFrame, compteurs: pd.DataFrame) -> pd.DataFrame:
    s = pd.DataFrame({"node_id": stations["station_id"], "type": "station", "axe": stations["axe"],
                      "sens": stations["sens"], "pk": stations["pk"], "delai_min": 0.0})
    source = compteurs.get("source", pd.Series("avatar", index=compteurs.index))
    c = pd.DataFrame({"node_id": compteurs["compteur_id"], "type": np.where(source == "here", "troncon", "compteur"),
                      "axe": compteurs["axe_rattache"],
                      "sens": compteurs["sens"], "pk": compteurs["pk_projete"],
                      "delai_min": compteurs.get("delai_rattachement_min", pd.Series(10.0, index=compteurs.index))})
    return pd.concat([s, c], ignore_index=True)  # stations d'abord, puis compteurs


def construire_aretes(noeuds: pd.DataFrame, vitesse_kmh=110.0) -> pd.DataFrame:
    """Construit le graphe à partir des seuls (axe, sens, pk). À remplacer par un vrai graphe
    (OSM, BD TOPO, Route 500) si des noeuds sont sur des axes qui se croisent."""
    aretes = []
    # une station de sens inconnu (0) est rattachée aux deux sens
    double = noeuds[noeuds["sens"].astype(int) == 0]
    noeuds = pd.concat([noeuds[noeuds["sens"].astype(int) != 0], double.assign(sens=1), double.assign(sens=2)])
    for (axe, sens), g in noeuds.groupby(["axe", "sens"]):
        signe = 1 if int(sens) == 1 else -1
        st = g[g["type"] != "compteur"].assign(o=lambda x: signe * x["pk"]).sort_values("o")
        for a, b in zip(st.itertuples(), st.iloc[1:].itertuples()):
            aretes.append((a.node_id, b.node_id, abs(b.pk - a.pk) / vitesse_kmh * 60, "axe"))
        for c in g[g["type"] == "compteur"].itertuples():
            aval = st[st["o"] >= signe * c.pk]
            if len(aval):
                b = aval.iloc[0]
                aretes.append((c.node_id, b["node_id"], abs(b["pk"] - c.pk) / vitesse_kmh * 60 + c.delai_min,
                               "rattachement"))
    return pd.DataFrame(aretes, columns=["source", "cible", "temps_min", "type"])


def matrice_temps(noeuds: pd.DataFrame, aretes: pd.DataFrame) -> np.ndarray:
    """temps[i, j] = temps de parcours (min) pour aller du noeud i au noeud j ; inf si j n'est pas en aval de i."""
    pos = {n: i for i, n in enumerate(noeuds["node_id"])}
    a = aretes[aretes["source"].isin(pos) & aretes["cible"].isin(pos)]
    n = len(noeuds)
    m = csr_matrix((a["temps_min"].values + 1e-6, (a["source"].map(pos), a["cible"].map(pos))), shape=(n, n))
    return shortest_path(m, directed=True, method="D")


def adjacences(temps: np.ndarray, sigma_min=60.0, tmax_min=240.0):
    """Noyau gaussien sur le temps de parcours (comme DCRNN). Renvoie deux matrices normalisées par ligne :
    A_amont[j, i] = poids du noeud amont i pour le noeud j (le trafic arrive de l'amont),
    A_aval = idem dans l'autre sens (une station saturée en aval peut reporter la demande vers l'amont)."""
    w = np.exp(-(temps / sigma_min) ** 2)
    w[(temps > tmax_min) | ~np.isfinite(temps)] = 0
    np.fill_diagonal(w, 0)
    amont = w.T  # ligne j : poids de chaque noeud i qui alimente j
    aval = w

    def norm(m):
        s = m.sum(1, keepdims=True)
        return np.divide(m, s, out=np.zeros_like(m), where=s > 0)

    return norm(amont).astype(np.float32), norm(aval).astype(np.float32)


def decaler(x: np.ndarray, k: int) -> np.ndarray:
    """x[t - k] avec NaN au début (k >= 0)."""
    out = np.full_like(x, np.nan, dtype=float)
    if k == 0:
        return x.astype(float).copy()
    out[k:] = x[:-k]
    return out


def variables_propagation(debit_norm: np.ndarray, temps_cs: np.ndarray, pas_min: float,
                          tau0_min=90.0, tmax_min=300.0, fenetres=(0, 1, 2, 3)) -> dict[str, np.ndarray]:
    """
    debit_norm : [T, C] débit des compteurs normalisé (1 = trafic moyen du compteur)
    temps_cs   : [C, S] temps de parcours compteur -> station (inf si pas en amont)

    Pour chaque station s et chaque pas t :
      prop_k = moyenne pondérée sur les compteurs amont c de debit_c(t - retard_cs - k),
               poids exp(-temps_cs / tau0) : les compteurs proches comptent davantage.
    Le retard retard_cs = temps_cs / pas : c'est ce qui encode la propagation de la vague de trafic.
    Renvoie un dict nom -> [T, S].
    """
    T, C = debit_norm.shape
    S = temps_cs.shape[1]
    out = {}
    for k in fenetres:
        num = np.zeros((T, S))
        den = np.zeros((T, S))
        for c in range(C):
            for s in range(S):
                tcs = temps_cs[c, s]
                if not np.isfinite(tcs) or tcs > tmax_min:
                    continue
                x = decaler(debit_norm[:, c], int(round(tcs / pas_min)) + k)
                w = np.exp(-tcs / tau0_min)
                ok = ~np.isnan(x)
                num[ok, s] += w * x[ok]
                den[ok, s] += w
        out[f"trafic_amont_propage_t-{k}"] = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    # compteur amont le plus proche, au bon retard
    plus_proche = np.full((T, S), np.nan)
    for s in range(S):
        t = temps_cs[:, s]
        if np.isfinite(t).any() and np.nanmin(t) <= tmax_min:
            c = int(np.nanargmin(t))
            plus_proche[:, s] = decaler(debit_norm[:, c], int(round(t[c] / pas_min)))
    out["trafic_compteur_amont_proche"] = plus_proche
    return out


def variables_aval(x: np.ndarray, temps_sc: np.ndarray, tmax_min=60.0) -> np.ndarray:
    """
    x        : [T, C] une variable de trafic (congestion d'un tronçon HERE, par exemple)
    temps_sc : [S, C] temps de parcours station -> noeud de trafic (inf si le noeud n'est pas en aval)
    Valeur au même instant du noeud de trafic le plus proche EN AVAL de chaque station : un bouchon
    juste après une aire pousse à s'y arrêter. NaN si aucun noeud en aval à moins de tmax_min.
    """
    T, S = x.shape[0], temps_sc.shape[0]
    out = np.full((T, S), np.nan)
    for s in range(S):
        t = temps_sc[s]
        if np.isfinite(t).any() and np.nanmin(t) <= tmax_min:
            out[:, s] = x[:, int(np.nanargmin(t))]
    return out
