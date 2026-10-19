"""
Prépare l'historique Qualicharge pour l'apprentissage, sur le poste qui a les exports mensuels
(charging_station_availability_snapshots_07.csv, _08, _09..., 1,2 à 1,6 Go chacun).

Deux opérations, indépendantes :
  1. --compresser : écrit chaque export .csv en .csv.xz à côté (6 à 10 fois plus petit), en flux,
     sans tout charger en mémoire. Le .csv d'origine est gardé, sauf avec --supprimer-csv.
  2. conversion : lit les exports (.csv ou .csv.xz) par morceaux, ne garde que les stations de
     stations.csv, met bout à bout tous les mois (un état qui chevauche deux fichiers reste continu),
     puis convertit au pas horaire avec donnees.convertir_snapshots_qualicharge.
     Sortie : occupation_bornes.csv (ou .csv.xz si le nom se termine par .xz).

Usage (PowerShell, depuis le dossier du dépôt ; en cmd, remplacer ` par ^) :
    python preparer_qualicharge.py --compresser `
        --snapshots "C:/Users/corde/OneDrive - ifpen.fr/Documents/TECHNIQUES/data_IRVE/Qualicharge/charging_station_availability_snapshots_*.csv"
    python preparer_qualicharge.py `
        --snapshots "C:/Users/corde/OneDrive - ifpen.fr/Documents/TECHNIQUES/data_IRVE/Qualicharge/charging_station_availability_snapshots_*.csv.xz" `
        --stations donnees_reelles/stations.csv --sortie donnees_reelles/occupation_bornes.csv.xz
"""
import argparse
import glob
import lzma
import shutil
import time
from pathlib import Path

import pandas as pd

from donnees import convertir_snapshots_qualicharge

COLONNES = ["station_key", "fetched_at", "last_seen_at", "latest_status_at",
            "availability.pdcs_with_dynamic_count", "availability.en_service_count", "availability.libre_count",
            "availability.occupied_count", "availability.reserved_count"]


def nombre(n):
    return f"{n:,}".replace(",", " ")


def lister(motifs):
    fichiers = sorted({f for m in motifs for f in glob.glob(m)})
    if not fichiers:
        raise SystemExit(f"aucun fichier pour {motifs}")
    return [Path(f) for f in fichiers]


def compresser(chemin: Path, niveau=6, supprimer=False):
    """chemin.csv -> chemin.csv.xz, en flux par blocs de 16 Mo."""
    cible = chemin.with_name(chemin.name + ".xz")
    if cible.exists() and cible.stat().st_size > 0:
        print(f"  {cible.name} existe déjà")
        return cible
    t0 = time.time()
    tmp = cible.with_name(cible.name + ".partiel")
    with open(chemin, "rb") as src, lzma.open(tmp, "wb", preset=niveau) as dst:
        shutil.copyfileobj(src, dst, 16 << 20)
    tmp.replace(cible)  # le .xz n'apparaît qu'une fois complet
    print(f"  {chemin.name} : {chemin.stat().st_size / 1e9:.2f} Go -> {cible.stat().st_size / 1e6:.0f} Mo "
          f"en {(time.time() - t0) / 60:.1f} min", flush=True)
    if supprimer:
        chemin.unlink()
    return cible


def lire_snapshots(fichiers, stations, morceau=1_000_000):
    """Ne garde que les stations voulues, fichier par fichier et morceau par morceau."""
    parts = []
    for f in fichiers:
        n, gardees = 0, 0
        for bloc in pd.read_csv(f, usecols=COLONNES, chunksize=morceau):
            n += len(bloc)
            bloc = bloc[bloc["station_key"].isin(stations)]
            gardees += len(bloc)
            parts.append(bloc)
        print(f"  {f.name} : {nombre(n)} lignes, {nombre(gardees)} pour les stations retenues", flush=True)
    snap = pd.concat(parts, ignore_index=True)
    # le même état peut figurer dans deux exports successifs
    return snap.drop_duplicates(subset=["station_key", "fetched_at"])


def controler(occ: pd.DataFrame):
    """Trous de collecte et part de statut inconnu, à noter dans .notes/inputs.md."""
    h = pd.to_datetime(occ["horodatage"])
    grille = pd.date_range(h.min(), h.max(), freq="1h")
    par_heure = occ.groupby(h)["station_id"].nunique().reindex(grille, fill_value=0)
    trous = par_heure[par_heure < 0.5 * par_heure.median()]
    connu = (occ["nb_pdc_statut_connu"] / occ["nb_pdc"].where(occ["nb_pdc"] > 0)).mean()
    print(f"\nPériode {h.min():%Y-%m-%d %H:%M} -> {h.max():%Y-%m-%d %H:%M}, {occ['station_id'].nunique()} stations, "
          f"{nombre(len(occ))} lignes station x heure")
    print(f"Part des PDC au statut connu : {connu:.1%}")
    print(f"Heures où moins de la moitié des stations remontent : {len(trous)}")
    if len(trous):
        jours = trous.groupby(trous.index.date).size()
        print("  par jour :", ", ".join(f"{j} ({n} h)" for j, n in jours.items()))
    nb = occ.groupby("station_id")["nb_pdc"].agg(["min", "max"])
    change = nb[nb["max"] - nb["min"] >= 1]
    print(f"Stations dont le nombre de PDC a changé sur la période : {len(change)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", nargs="+", required=True, help="fichiers ou motifs (*.csv, *.csv.xz)")
    ap.add_argument("--compresser", action="store_true", help="seulement écrire les .csv.xz")
    ap.add_argument("--supprimer-csv", action="store_true", help="avec --compresser : supprimer le .csv une fois compressé")
    ap.add_argument("--niveau", type=int, default=6, help="niveau xz (0 à 9)")
    ap.add_argument("--stations", default="donnees_reelles/stations.csv")
    ap.add_argument("--sortie", default="donnees_reelles/occupation_bornes.csv.xz")
    ap.add_argument("--pas", default="1h")
    a = ap.parse_args()
    fichiers = lister(a.snapshots)

    if a.compresser:
        print(f"Compression de {len(fichiers)} fichier(s)")
        for f in fichiers:
            if f.suffix == ".csv":
                compresser(f, a.niveau, a.supprimer_csv)
        return

    stations = set(pd.read_csv(a.stations)["station_id"])
    print(f"Lecture de {len(fichiers)} export(s), {len(stations)} stations à garder")
    snap = lire_snapshots(fichiers, stations)
    print(f"Conversion au pas {a.pas} de {nombre(len(snap))} états", flush=True)
    occ = convertir_snapshots_qualicharge(snap, pas=a.pas)
    Path(a.sortie).parent.mkdir(parents=True, exist_ok=True)
    occ.to_csv(a.sortie, index=False)
    print(f"-> {a.sortie}")
    controler(occ)


if __name__ == "__main__":
    main()
