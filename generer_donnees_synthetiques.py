"""
Génère un jeu de données synthétique qui respecte le schéma d'entrée attendu
par `entrainer.py`, en attendant les vraies données.

Le scénario imite la réalité visée :
  - deux autoroutes au départ de Paris (A10 vers Bordeaux, A6 vers Lyon), dans les deux sens ;
  - le trafic AUTOROUTIER n'est pas mesuré (comme en vrai) : il n'existe que comme variable cachée ;
  - quelques compteurs type Avatar sur des routes nationales parallèles ou en entrée d'axe,
    qui voient une fraction bruitée du flux autoroutier ;
  - des tronçons HERE de 10 km SUR l'autoroute (format de here_flux.py) : congestion et jam factor,
    qui ne réagissent qu'au-delà de 70 % de la capacité, et un indice de volume (véhicules sondes),
    proportionnel au flux mais bruité ;
  - des stations de recharge sur les aires, dont l'occupation dépend du flux qui passe devant
    elles, avec le retard de propagation (une vague de départs parisienne arrive à pk 300
    environ 3 h plus tard) ;
  - calendrier 2026 : week-ends, jours fériés, vacances scolaires, jours de départ / retour.

Usage :
    python generer_donnees_synthetiques.py --sortie donnees_synthetiques --debut 2026-03-01 --mois 6
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

VITESSE_KMH = 110.0

# Périodes de vacances scolaires 2026 (approximation toutes zones confondues)
VACANCES_2026 = [
    ("2026-02-07", "2026-03-09"),
    ("2026-04-04", "2026-05-04"),
    ("2026-07-04", "2026-08-31"),
    ("2026-10-17", "2026-11-02"),
    ("2026-12-19", "2027-01-04"),
]
FERIES_2026 = ["2026-01-01", "2026-04-06", "2026-05-01", "2026-05-08", "2026-05-14",
               "2026-05-25", "2026-07-14", "2026-08-15", "2026-11-01", "2026-11-11", "2026-12-25"]

AXES = {
    # axe: (longueur km, lat/lon départ, lat/lon arrivée)
    "A10": (560, (48.70, 2.25), (44.90, -0.50)),
    "A6": (460, (48.65, 2.40), (45.75, 4.85)),
}


def calendrier(index: pd.DatetimeIndex) -> pd.DataFrame:
    jours = pd.DatetimeIndex(sorted(set(index.normalize())))
    cal = pd.DataFrame({"date": jours})
    cal["ferie"] = cal["date"].isin(pd.to_datetime(FERIES_2026)).astype(int)
    vac = np.zeros(len(cal), dtype=int)
    debut_vac = np.zeros(len(cal), dtype=int)
    fin_vac = np.zeros(len(cal), dtype=int)
    for d, f in VACANCES_2026:
        d, f = pd.Timestamp(d), pd.Timestamp(f)
        m = (cal["date"] >= d) & (cal["date"] < f)
        vac[m.values] = 1
        debut_vac[((cal["date"] >= d) & (cal["date"] < d + pd.Timedelta(days=2))).values] = 1
        fin_vac[((cal["date"] >= f - pd.Timedelta(days=2)) & (cal["date"] < f)).values] = 1
    cal["vacances"] = vac
    cal["debut_vacances"] = debut_vac
    cal["fin_vacances"] = fin_vac
    # Prévision Bison Futé officielle (si disponible en vrai, c'est une excellente variable)
    dow = cal["date"].dt.dayofweek
    couleur = np.zeros(len(cal), dtype=int)  # 0 vert, 1 orange, 2 rouge, 3 noir
    couleur[(dow == 4) | (dow == 6)] = 1
    couleur[(cal["vacances"] == 1) & (dow == 5)] = 2
    couleur[(cal["debut_vacances"] == 1) | (cal["fin_vacances"] == 1)] = 2
    couleur[(cal["date"].dt.month.isin([7, 8])) & (dow == 5) & (cal["debut_vacances"] + (cal["date"].dt.day <= 2) > 0)] = 3
    cal["bison_fute_couleur"] = couleur
    return cal


def profil_demande(index, cal, sens, rng):
    """Flux (véh/h) qui entre sur l'axe à son origine, pour un sens donné."""
    h = index.hour.values + index.minute.values / 60
    dow = index.dayofweek.values
    c = cal.set_index("date").reindex(index.normalize())
    vac, deb, fin, fer = (c[k].values for k in ("vacances", "debut_vacances", "fin_vacances", "ferie"))

    base = 600 + 1500 * np.exp(-((h - 9) / 2.5) ** 2) + 1800 * np.exp(-((h - 17.5) / 2.5) ** 2)
    base *= np.where(h < 5, 0.3, 1.0)
    f = np.ones_like(base)
    f *= 1 + 0.25 * vac + 0.35 * (index.month.isin([7, 8]))
    if sens == 1:  # aller : départs en vacances
        f *= 1 + 0.35 * (dow == 4) * (h > 13)
        f *= 1 + 0.7 * vac * (dow == 5) * np.exp(-((h - 10) / 3) ** 2)
        f *= 1 + 0.6 * deb + 0.2 * fer
    else:  # retour
        f *= 1 + 0.6 * (dow == 6) * np.exp(-((h - 17) / 3) ** 2)
        f *= 1 + 0.6 * fin * np.exp(-((h - 16) / 4) ** 2)
    bruit = np.exp(np.convolve(rng.normal(0, 0.08, len(h)), np.ones(6) / 6, mode="same"))
    # aléa journalier non prévisible par le calendrier (météo, événement, grève...) :
    # c'est précisément ce que les compteurs voient et que le calendrier seul ne voit pas
    jours = index.normalize()
    alea = pd.Series(np.exp(rng.normal(0, 0.18, len(jours.unique()))), index=jours.unique())
    return base * f * bruit * alea.reindex(jours).values


def interp_latlon(axe, pk):
    L, (la0, lo0), (la1, lo1) = AXES[axe]
    r = pk / L
    return la0 + r * (la1 - la0), lo0 + r * (lo1 - lo0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sortie", default="donnees_synthetiques")
    ap.add_argument("--debut", default="2026-03-01")
    ap.add_argument("--mois", type=int, default=6)
    ap.add_argument("--pas", default="1h", help="pas de temps (1h, 30min, 15min)")
    ap.add_argument("--graine", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.graine)
    out = Path(args.sortie)
    out.mkdir(parents=True, exist_ok=True)

    debut = pd.Timestamp(args.debut)
    index = pd.date_range(debut, debut + pd.DateOffset(months=args.mois), freq=args.pas, inclusive="left")
    pas_h = (index[1] - index[0]) / pd.Timedelta(hours=1)
    cal = calendrier(index)
    T = len(index)

    stations, compteurs, occ_rows, cpt_rows, here_rows = [], [], [], [], []
    for axe, (L, _, _) in AXES.items():
        for sens in (1, 2):
            demande = profil_demande(index, cal, sens, rng)

            def flux_latent(pk):
                """Flux autoroutier caché au pk donné : la demande d'origine, retardée du temps de trajet."""
                dist = pk if sens == 1 else L - pk
                retard = int(round(dist / VITESSE_KMH / pas_h))
                f = np.roll(demande, retard)
                f[:retard] = demande[0]
                return f * (1 - 0.0006 * dist) + rng.normal(0, 40, T)

            # Stations sur aires, environ tous les 60 km
            for k, pk in enumerate(np.arange(40, L - 20, 60) + rng.integers(-10, 10)):
                sid = f"ST_{axe}_{sens}_{k:02d}"
                nb_pdc = int(rng.choice([4, 6, 8, 10, 12]))
                lat, lon = interp_latlon(axe, pk)
                stations.append(dict(station_id=sid, nom=f"Aire {axe} pk{pk}", lat=round(lat, 5), lon=round(lon, 5),
                                     axe=axe, sens=sens, pk=float(pk), nb_pdc=nb_pdc,
                                     puissance_kw=int(rng.choice([150, 300]))))
                F = flux_latent(pk)
                dist = pk if sens == 1 else L - pk
                besoin = np.clip((dist - 80) / 200, 0.05, 1.0)  # on recharge rarement juste après le départ
                part_ve = 0.035 * (1 + 0.15 * np.arange(T) / T)
                lam = F * part_ve * besoin * 0.08  # véhicules qui veulent charger sur cette aire
                rho = 1 - np.exp(-lam / nb_pdc * 1.3)
                hs = np.zeros(T, dtype=int)  # pannes : quelques épisodes de PDC hors service
                for _ in range(rng.integers(0, 4)):
                    d0 = rng.integers(0, T)
                    hs[d0:d0 + rng.integers(6, 72)] += 1
                hs = np.minimum(hs, nb_pdc - 1)
                # moyenne de 4 relevés par pas (statuts toutes les 15 min agrégés à l'heure)
                occ = rng.binomial(nb_pdc - hs, np.clip(rho, 0, 1), size=(4, T)).mean(0)
                occ_rows.append(pd.DataFrame({"horodatage": index, "station_id": sid, "nb_pdc": nb_pdc,
                                              "nb_pdc_occupes": occ, "nb_pdc_hors_service": hs}))

            # Compteurs Avatar : routes parallèles ou entrées d'axe, jamais sur l'autoroute elle-même
            pks_cpt = [5] + list(rng.choice(np.arange(60, L - 60, 20), size=2, replace=False))
            for k, pk in enumerate(sorted(pks_cpt)):
                cid = f"CPT_{axe}_{sens}_{k}"
                lat, lon = interp_latlon(axe, pk)
                compteurs.append(dict(compteur_id=cid, lat=round(lat + 0.05, 5), lon=round(lon + 0.05, 5),
                                      route=f"N{rng.integers(10, 150)}", axe_rattache=axe, sens=sens,
                                      pk_projete=float(pk), delai_rattachement_min=float(rng.integers(5, 20))))
                F = flux_latent(pk)
                h = index.hour.values
                local = 250 * (np.exp(-((h - 8) / 1.5) ** 2) + np.exp(-((h - 18) / 1.5) ** 2))
                mes = np.maximum(0, rng.uniform(0.15, 0.4) * F + local + rng.normal(0, 60, T))
                manquant = rng.random(T) < 0.02  # trous de mesure, comme en vrai
                mes = np.where(manquant, np.nan, np.round(mes))
                cpt_rows.append(pd.DataFrame({"horodatage": index, "compteur_id": cid, "debit_vh": mes}))

            # Tronçons HERE de 10 km : vitesse et congestion (seuil de capacité), indice de volume (sondes)
            capacite = 4200.0
            for pk0 in np.arange(0, L - 5, 10.0):
                F = np.maximum(flux_latent(pk0 + 5), 0)
                charge = F / capacite
                cong = np.clip(1.6 * (charge - 0.7), 0, 0.85) ** 1.3 + np.abs(rng.normal(0, 0.02, T))
                cong = np.clip(cong, 0, 0.9)
                vf = 125.0
                here_rows.append(pd.DataFrame({
                    "horodatage": index, "axe": axe, "sens": sens, "troncon_pk": pk0, "longueur_couverte_km": 10.0,
                    "nb_segments": 8, "vitesse_kmh": np.round(vf * (1 - cong), 1), "vitesse_fluide_kmh": vf,
                    "congestion": np.round(cong, 3), "temps_parcours_min": np.round(600 / (vf * (1 - cong)), 2),
                    "jam_factor": np.round(np.clip(10 * cong / 0.9 + rng.normal(0, 0.2, T), 0, 10), 2),
                    "confiance": np.round(rng.uniform(0.7, 1.0, T), 2),
                    "indice_volume": rng.poisson(F * 0.02)}))

    pd.DataFrame(stations).to_csv(out / "stations.csv", index=False)
    (out / "here").mkdir(exist_ok=True)
    pd.concat(here_rows).to_csv(out / "here" / "here_troncons.csv", index=False)
    pd.DataFrame(compteurs).to_csv(out / "compteurs.csv", index=False)
    pd.concat(occ_rows).to_csv(out / "occupation_bornes.csv", index=False)
    pd.concat(cpt_rows).to_csv(out / "comptages.csv", index=False)
    cal.to_csv(out / "calendrier.csv", index=False)
    print(f"Écrit dans {out}/ : {len(stations)} stations, {len(compteurs)} compteurs, "
          f"{sum(len(h) for h in here_rows) // T} tronçons HERE, {T} pas de temps")


if __name__ == "__main__":
    main()
