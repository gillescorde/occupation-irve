"""
Téléchargement OpenStreetMap (Overpass) des voies des axes, des échangeurs et des aires, pour
`traces_osm.py`. À lancer depuis un poste qui accède à Overpass (bureau, ou environnement cloud
autorisé). Une requête à la fois, avec réessais : Overpass limite la charge par adresse.

Usage :
    python telecharger_osm.py --sortie donnees_reelles/osm/brut [--serveur https://overpass-api.de/api/interpreter]
    python traces_osm.py --brut donnees_reelles/osm/brut --sortie donnees_reelles
"""
import argparse
import time
from pathlib import Path

import requests

from reperer_stations_autoroute import AXES

ZONE = 'area["ISO3166-1"="FR"][admin_level=2]->.fr;'


def requete_axe(axe):
    lettre, num = axe[0], axe[1:]
    refs = [f"{lettre} ?{num}"] + (["A ?430"] if axe == "N90" else [])
    motif = "|".join(refs)
    return (f'[out:json][timeout:600];{ZONE}'
            f'way(area.fr)["highway"~"^(motorway|trunk)$"]["ref"~"(^|;)\\\\s*({motif})\\\\s*($|;)"];out geom;')


REQ_ECHANGEURS = f'[out:json][timeout:600];{ZONE}node(area.fr)["highway"="motorway_junction"];out;'
REQ_AIRES = f'[out:json][timeout:600];{ZONE}nwr(area.fr)["highway"~"^(services|rest_area)$"];out center;'


def telecharger(serveur, requete, chemin, essais=5):
    if chemin.exists() and chemin.stat().st_size > 100:
        return "déjà là"
    for i in range(essais):
        try:
            r = requests.post(serveur, data={"data": requete}, timeout=900)
            if r.status_code in (429, 504):
                time.sleep(30 * (i + 1))
                continue
            r.raise_for_status()
            chemin.write_bytes(r.content)
            return f"{len(r.content) / 1e6:.1f} Mo"
        except requests.RequestException as e:
            err = e
            time.sleep(10 * (i + 1))
    return f"échec ({err if 'err' in dir() else 'quota'})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sortie", default="donnees_reelles/osm/brut")
    ap.add_argument("--serveur", default="https://overpass-api.de/api/interpreter")
    a = ap.parse_args()
    out = Path(a.sortie)
    out.mkdir(parents=True, exist_ok=True)
    taches = [(f"axe_{axe}.json", requete_axe(axe)) for axe in AXES]
    taches += [("echangeurs.json", REQ_ECHANGEURS), ("aires.json", REQ_AIRES)]
    for nom, req in taches:
        print(f"{nom:20s} {telecharger(a.serveur, req, out / nom)}", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()
