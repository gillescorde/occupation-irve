"""
Visuels des données d'entrée : occupation des bornes, débits Avatar, calendrier Bison Futé et
croisement trafic / occupation. Écrit des PNG dans le dossier de sortie.

Usage :
    python visuels.py --donnees donnees_reelles --sortie visuels
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

BLEU, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TEXTE, TEXTE2, GRILLE = "#0b0b0b", "#52514e", "#e4e3df"
BF = ["#5bb781", "#f49a3f", "#e6332a", "#1d1d1b"]  # couleurs officielles Bison Futé : vert, orange, rouge, noir

JOURS = ["lun", "mar", "mer", "jeu", "ven", "sam", "dim"]
MOIS = ["janv", "févr", "mars", "avr", "mai", "juin", "juil", "août", "sept", "oct", "nov", "déc"]
fmt_jour = matplotlib.ticker.FuncFormatter(lambda x, _: (lambda t: f"{JOURS[t.weekday()]} {t.day}")(mdates.num2date(x)))
fmt_mois = matplotlib.ticker.FuncFormatter(lambda x, _: MOIS[mdates.num2date(x).month - 1])

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRILLE, "axes.labelcolor": TEXTE2, "xtick.color": TEXTE2,
    "ytick.color": TEXTE2, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": TEXTE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRILLE,
    "grid.linewidth": 0.6, "lines.linewidth": 2, "legend.frameon": False, "figure.dpi": 130})


def occupation(d, out):
    occ = pd.read_csv(d / "occupation_bornes.csv", parse_dates=["horodatage"])
    st = pd.read_csv(d / "stations.csv")
    occ = occ[occ["nb_pdc_statut_connu"] > 0]
    occ["taux"] = (occ["nb_pdc_occupes"] / occ["nb_pdc_statut_connu"]).clip(0, 1) * 100
    occ["autoroute"] = occ["station_id"].isin(st["station_id"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    s = occ[occ["autoroute"]].groupby("horodatage")["taux"]
    a1.fill_between(s.quantile(0.25).index, s.quantile(0.25), s.quantile(0.75), color=BLEU, alpha=0.15, lw=0,
                    label="Moitié centrale des stations")
    a1.plot(s.mean().index, s.mean().values, color=BLEU, label="Moyenne")
    a1.axvspan(pd.Timestamp("2026-07-09 14:00"), pd.Timestamp("2026-07-09 23:00"), color="#f4f3f0", zorder=0)
    a1.text(pd.Timestamp("2026-07-09 18:30"), 2, "trou de\ncollecte", ha="center", color=TEXTE2, fontsize=8)
    a1.set_title(f"Occupation des {occ.loc[occ['autoroute'], 'station_id'].nunique()} stations d'autoroute, 9 au 15 juillet 2026")
    a1.set_ylabel("Taux d'occupation (%)")
    a1.xaxis.set_major_formatter(fmt_jour)
    a1.legend(loc="upper left")
    a1.set_ylim(0, None)
    # distribution par station : moyenne et pointe (P90) des stations d'autoroute
    g = occ[occ["autoroute"]].groupby("station_id")["taux"]
    m, p90 = g.mean(), g.quantile(0.9)
    a2.scatter(m, p90, s=18, color=BLEU, edgecolor="white", linewidth=0.6)
    a2.plot([0, 100], [0, 100], color=GRILLE, lw=1, zorder=0)
    for seuil, c in ((40, BF[1]), (70, BF[2]), (90, BF[3])):
        a2.axhline(seuil, color=c, lw=1, ls="--")
        a2.text(1, seuil + 1, f"{seuil} %", color=TEXTE2, fontsize=8)
    a2.set_xlim(0, 60); a2.set_ylim(0, 105)
    a2.set_title("Station par station (autoroute)")
    a2.set_xlabel("Occupation moyenne (%)"); a2.set_ylabel("Pointe : 90ᵉ centile (%)")
    fig.tight_layout()
    fig.savefig(out / "1_occupation.png")
    plt.close(fig)


def debits(d, out):
    cpt = pd.read_csv(d / "debits" / "compteurs.csv")
    inter = set(cpt.loc[cpt["hors_agglo_extremites"], "compteur_id"])
    q = pd.read_csv(d / "debits" / "comptages.csv", usecols=["horodatage", "compteur_id", "debit_vh", "part_estimee_pct"],
                    parse_dates=["horodatage"])
    q = q[q["compteur_id"].isin(inter) & (q["part_estimee_pct"] <= 50)]
    q["jour"] = q["horodatage"].dt.floor("D")
    j = q.groupby(["compteur_id", "jour"])["debit_vh"].agg(["sum", "size"])
    j = j[j["size"] >= 20]["sum"].unstack(0).sort_index()
    indice = (j / j.median()).median(axis=1)  # trafic journalier relatif, médiane sur les compteurs
    cal = pd.read_csv(d / "calendrier" / "calendrier.csv", parse_dates=["date"]).set_index("date")
    cal = cal.reindex(indice.index)
    fig, ax = plt.subplots(figsize=(13, 4.4))
    vac = cal["nb_zones_vac"].fillna(0) > 0
    for debut, fin in _plages(vac):
        ax.axvspan(debut, fin + pd.Timedelta(days=1), color="#eef3fb", zorder=0)
    ax.plot(indice.index, indice.values, color=BLEU, lw=1.6, label="Trafic journalier (médiane des compteurs)")
    niv = cal[["bf_depart_max", "bf_retour_max"]].max(axis=1).fillna(0).astype(int)
    for k in (1, 2, 3):
        x = niv.index[niv == k]
        ax.scatter(x, indice.reindex(x), s=34, color=BF[k], edgecolor="white", linewidth=0.8, zorder=3,
                   label=f"Jour {['', 'orange', 'rouge', 'noir'][k]} Bison Futé")
    ax.set_title(f"Débits Avatar : {len(j.columns)} compteurs interurbains, 1er mars au 17 octobre 2026")
    ax.set_ylabel("Trafic relatif (1 = jour médian)")
    ax.xaxis.set_major_formatter(fmt_mois)
    ax.text(0.995, 0.02, "fond bleu : vacances scolaires (au moins une zone)", transform=ax.transAxes,
            ha="right", color=TEXTE2, fontsize=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
    fig.tight_layout()
    fig.savefig(out / "2_debits_avatar.png")
    plt.close(fig)


def _plages(masque):
    plages, debut = [], None
    for jour, v in masque.items():
        if v and debut is None:
            debut = jour
        if not v and debut is not None:
            plages.append((debut, prec)); debut = None
        prec = jour
    if debut is not None:
        plages.append((debut, prec))
    return plages


def bison_fute(d, out):
    cal = pd.read_csv(d / "calendrier" / "calendrier.csv", parse_dates=["date"])
    cal["an"], cal["doy"] = cal["date"].dt.year, cal["date"].dt.dayofyear
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.2), sharex=True)
    for ax, sens, titre in ((axes[0], "bf_depart_max", "Départs"), (axes[1], "bf_retour_max", "Retours")):
        m = cal.pivot_table(index="an", columns="doy", values=sens).reindex(columns=range(1, 367))
        ax.imshow(np.ma.masked_invalid(m.values), aspect="auto", cmap=ListedColormap(BF), vmin=0, vmax=3,
                  interpolation="nearest", extent=[0.5, 366.5, m.index.max() + 0.5, m.index.min() - 0.5])
        ax.set_yticks(m.index); ax.grid(False)
        ax.set_title(f"Bison Futé, {titre.lower()} : niveau le plus élevé toutes zones confondues")
        ax.set_facecolor("#f4f3f0")
    debuts = pd.date_range("2025-01-01", "2025-12-01", freq="MS")
    axes[1].set_xticks(debuts.dayofyear)
    axes[1].set_xticklabels([x.strftime("%b") for x in debuts])
    axes[0].text(1.0, 1.02, "2020 : calendrier introuvable", transform=axes[0].transAxes, ha="right",
                 color=TEXTE2, fontsize=8)
    poignees = [plt.Rectangle((0, 0), 1, 1, color=c) for c in BF]
    fig.legend(poignees, ["vert", "orange", "rouge", "noir"], loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "3_bison_fute.png")
    plt.close(fig)


def croisement(d, out):
    occ = pd.read_csv(d / "occupation_bornes.csv", parse_dates=["horodatage"])
    occ = occ[occ["nb_pdc_statut_connu"] > 0]
    occ["taux"] = (occ["nb_pdc_occupes"] / occ["nb_pdc_statut_connu"]).clip(0, 1) * 100
    st = pd.read_csv(d / "stations.csv").set_index("station_id")
    liens = pd.read_csv(d / "rattachement_interurbains" / "liens_compteur_station.csv")
    liens = liens[liens["station_id"].isin(occ["station_id"].unique())]
    meilleurs = liens.sort_values("r_brut", ascending=False).drop_duplicates("station_id").head(3)
    cpt = pd.read_csv(d / "debits" / "compteurs.csv").set_index("compteur_id")
    q = pd.read_csv(d / "debits" / "comptages.csv", usecols=["horodatage", "compteur_id", "debit_vh"],
                    parse_dates=["horodatage"])
    q = q[q["compteur_id"].isin(meilleurs["compteur_id"]) & (q["horodatage"] >= "2026-07-09")
          & (q["horodatage"] < "2026-07-15 12:00")]
    fig, axes = plt.subplots(len(meilleurs), 1, figsize=(13, 3.0 * len(meilleurs)), sharex=True)
    for ax, r in zip(np.atleast_1d(axes), meilleurs.itertuples()):
        o = occ[occ["station_id"] == r.station_id].set_index("horodatage")["taux"]
        c = q[q["compteur_id"] == r.compteur_id].set_index("horodatage")["debit_vh"].shift(int(r.retard_h))
        ax.plot(o.index, o.values, color=BLEU, label="Occupation de la station (%)")
        cn = c / c.max() * max(o.max(), 1)
        ax.plot(cn.index, cn.values, color=ORANGE, lw=1.4,
                label=f"Débit du compteur amont, décalé de {int(r.retard_h)} h (mis à l'échelle)")
        s = st.loc[r.station_id]
        ax.set_title(f"{s['nom'][:60]} ({s['axe']}, pk {s['pk']:.0f}) ; compteur {cpt.loc[r.compteur_id, 'route']} "
                     f"à {r.temps_min:.0f} min, r = {r.r_brut:.2f}", fontsize=10)
        ax.set_ylim(0, None)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2)
    axes[-1].xaxis.set_major_formatter(fmt_jour)
    fig.suptitle("Trafic amont et occupation : les trois stations les mieux corrélées", fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "4_trafic_occupation.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donnees", default="donnees_reelles")
    ap.add_argument("--sortie", default="visuels")
    a = ap.parse_args()
    d, out = Path(a.donnees), Path(a.sortie)
    out.mkdir(parents=True, exist_ok=True)
    for f in (occupation, bison_fute, croisement, debits):
        f(d, out)
        print(f.__name__, "ok", flush=True)


if __name__ == "__main__":
    main()
