"""
Scénarios de pénétration des véhicules électriques (VE) : « que devient l'occupation des aires si la
part de VE dans le trafic passe de 3 % à 10 % ? ».

Pourquoi un module à part au lieu d'une variable d'entrée du modèle : sur l'historique, la part de VE
vaut ~3 % partout et tout le temps. Un modèle appris ne peut rien apprendre d'une variable constante,
et un gradient boosting n'extrapole pas au-delà de ce qu'il a vu. On sépare donc :

  1. ce que le modèle apprend : la charge offerte à chaque station, c'est-à-dire le nombre moyen de
     véhicules qui voudraient être en charge en même temps (en erlangs) ;
  2. ce que le scénario change : cette charge est proportionnelle au flux de VE, donc à la pénétration p,
         A'(s, t) = A(s, t) · (p' / p_ref) · facteur_besoin ;
  3. une loi de files d'attente (Erlang B, c = nombre de PDC) qui convertit la charge en occupation,
     avec la saturation : une station de 8 PDC ne dépasse jamais 100 %, mais la probabilité de la
     trouver pleine grimpe vite.

Occupation et charge sont liées par la charge écoulée (Erlang B) :
    y = A · (1 − B(A, c)) / c,     B(A, c) = probabilité que les c PDC soient tous occupés.
Cette relation est inversée pour retrouver A à partir d'une occupation observée ou prévue.

Usage :
    python scenarios.py --occupation donnees_reelles/occupation_bornes.csv \\
        --penetration-ref 0.03 --penetration 0.10 --sortie scenario_10pc.csv
    # sur les prévisions d'un modèle (colonne « prevu ») :
    python scenarios.py --predictions resultats/predictions_test.csv --stations donnees_reelles/stations.csv \\
        --modele hgb_trafic_prevu --penetration 0.10 --pdc-supplementaires 4
"""
import argparse

import numpy as np
import pandas as pd

SEUILS_COULEUR = [0.40, 0.70, 0.90]
COULEURS = np.array(["vert", "orange", "rouge", "noir"])


def erlang_b(A, c):
    """Probabilité de blocage d'Erlang B, vectorisée. A : charge offerte (erlangs), c : nombre de PDC (entier >= 1).
    Récurrence stable : B(0) = 1, B(k) = A·B(k−1) / (k + A·B(k−1))."""
    A = np.asarray(A, dtype=float)
    c = np.asarray(c, dtype=int)
    A, c = np.broadcast_arrays(A, c)
    B = np.ones_like(A)
    for k in range(1, int(c.max()) + 1):
        Bk = A * B / (k + A * B)
        B = np.where(k <= c, Bk, B)
    return B


def occupation_depuis_charge(A, c):
    """Taux d'occupation moyen = charge écoulée / nombre de PDC."""
    c = np.maximum(np.asarray(c, dtype=int), 1)
    return np.asarray(A) * (1 - erlang_b(A, c)) / c


def charge_depuis_occupation(y, c, y_max=0.97, iterations=60):
    """Inverse de occupation_depuis_charge par dichotomie (la fonction est croissante en A).
    Au-delà de y_max, l'inversion n'a plus de sens : la station est saturée et la charge réelle
    n'est plus observable. On plafonne, ce qui sous-estime la charge des stations déjà pleines."""
    y = np.clip(np.asarray(y, dtype=float), 0, y_max)
    c = np.maximum(np.asarray(c, dtype=int), 1)
    lo, hi = np.zeros_like(y), np.full_like(y, 1.0) * 60 * c
    for _ in range(iterations):
        mid = (lo + hi) / 2
        trop = occupation_depuis_charge(mid, c) > y
        hi = np.where(trop, mid, hi)
        lo = np.where(trop, lo, mid)
    return (lo + hi) / 2


def simuler(df: pd.DataFrame, penetration_ref: float, penetration: float, facteur_besoin=1.0,
            pdc_supplementaires=0, facteur_pdc=1.0) -> pd.DataFrame:
    """df : colonnes taux_ref (occupation de référence, 0..1) et nb_pdc. Renvoie df complété de
    charge_offerte, taux_scenario, proba_station_pleine (ref et scénario) et couleurs."""
    out = df.copy()
    c_ref = np.maximum(np.round(out["nb_pdc"]).astype(int), 1)
    A = charge_depuis_occupation(out["taux_ref"].values, c_ref.values)
    c_new = np.maximum(np.round(c_ref * facteur_pdc + pdc_supplementaires).astype(int), 1)
    A_new = A * penetration / penetration_ref * facteur_besoin
    out["charge_offerte_ref"] = A.round(3)
    out["charge_offerte_scenario"] = A_new.round(3)
    out["nb_pdc_scenario"] = c_new
    out["taux_scenario"] = occupation_depuis_charge(A_new, c_new.values).round(4)
    out["proba_pleine_ref"] = erlang_b(A, c_ref.values).round(4)
    out["proba_pleine_scenario"] = erlang_b(A_new, c_new.values).round(4)
    out["couleur_ref"] = COULEURS[np.digitize(out["taux_ref"].values, SEUILS_COULEUR)]
    out["couleur_scenario"] = COULEURS[np.digitize(out["taux_scenario"].values, SEUILS_COULEUR)]
    return out


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--occupation", help="occupation_bornes.csv (observé)")
    src.add_argument("--predictions", help="predictions_test.csv d'entrainer.py (colonne prevu)")
    ap.add_argument("--stations", help="stations.csv (nb_pdc), requis avec --predictions")
    ap.add_argument("--modele", default=None, help="avec --predictions : modèle à utiliser (colonne modele)")
    ap.add_argument("--penetration-ref", type=float, default=0.03, help="part de VE dans le trafic de l'historique")
    ap.add_argument("--penetration", type=float, required=True, help="part de VE du scénario (ex. 0.10)")
    ap.add_argument("--facteur-besoin", type=float, default=1.0,
                    help="recharges par VE relativement à aujourd'hui (<1 si les batteries 2030 sont plus grandes)")
    ap.add_argument("--pdc-supplementaires", type=int, default=0, help="PDC ajoutés à chaque station")
    ap.add_argument("--facteur-pdc", type=float, default=1.0, help="multiplie le nombre de PDC de chaque station")
    ap.add_argument("--sortie", default="scenario.csv")
    args = ap.parse_args()

    if args.occupation:
        o = pd.read_csv(args.occupation, parse_dates=["horodatage"])
        o["taux_ref"] = o["nb_pdc_occupes"] / o["nb_pdc_statut_connu"].where(o["nb_pdc_statut_connu"] > 0)
        o["nb_pdc"] = o["nb_pdc_statut_connu"]
        df = o.dropna(subset=["taux_ref"])[["horodatage", "station_id", "nb_pdc", "taux_ref"]]
    else:
        if not args.stations:
            raise SystemExit("--stations est requis avec --predictions")
        p = pd.read_csv(args.predictions, parse_dates=["horodatage"])
        if args.modele:
            p = p[p["modele"] == args.modele]
        st = pd.read_csv(args.stations)[["station_id", "nb_pdc"]]
        df = p.rename(columns={"prevu": "taux_ref"}).merge(st, on="station_id")[
            ["horodatage", "station_id", "nb_pdc", "taux_ref"] + (["modele"] if "modele" in p else [])]

    res = simuler(df, args.penetration_ref, args.penetration, args.facteur_besoin,
                  args.pdc_supplementaires, args.facteur_pdc)
    res.to_csv(args.sortie, index=False)

    def resume(col_taux, col_pleine, col_couleur):
        return {"occupation moyenne": f"{res[col_taux].mean():.1%}",
                "heures rouge ou noir": f"{res[col_couleur].isin(['rouge', 'noir']).mean():.1%}",
                "proba moyenne station pleine": f"{res[col_pleine].mean():.1%}"}

    print(f"Référence (pénétration {args.penetration_ref:.0%}) :", resume("taux_ref", "proba_pleine_ref", "couleur_ref"))
    print(f"Scénario  (pénétration {args.penetration:.0%}, besoin x{args.facteur_besoin}, "
          f"PDC x{args.facteur_pdc} +{args.pdc_supplementaires}) :",
          resume("taux_scenario", "proba_pleine_scenario", "couleur_scenario"))
    print(f"-> {args.sortie}")


if __name__ == "__main__":
    main()
