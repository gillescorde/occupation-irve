"""
Extraction des couleurs Bison Futé depuis les calendriers annuels PDF (bison-fute.gouv.fr).

Dans ces calendriers, chaque mois est une colonne ; chaque jour a deux cellules (départ, retour).
Une cellule se lit ainsi :
  - la couleur du fond (bords de la cellule) est le niveau national ;
  - un cartouche central portant des numéros de zone (« 1-3 » = zones 1 et 3, pas 1 à 3) donne
    le niveau, plus élevé, de ces zones.
Zones : 1 Île-de-France, 2 Grand-Ouest et Nord, 3 Bourgogne et Est, 4 Auvergne-Rhône-Alpes,
5 Sud-Ouest, 6 Arc méditerranéen.

Les couleurs sont des aplats graphiques : on rend la page en image et on lit les pixels au bord de
la cellule et à côté des numéros de zone, puis on classe la teinte en vert, orange, rouge, noir.

Sortie : une ligne par (date, sens, zone) non verte ; zone 0 = niveau national.
    date,sens,zone,couleur
Usage :
    python bison_fute_pdf.py brut/bison_fute/bison_fute_2026.pdf [...] --sortie bison_fute.csv
    # avec le CSV 2016 au format openeventdatabase (codes O/R/N) :
    python bison_fute_pdf.py brut/bison_fute/*.pdf --csv-2016 brut/bison_fute/bison_fute_2016.csv --sortie bison_fute.csv
"""
import argparse
import re

import numpy as np
import pandas as pd

MOIS = {"janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6, "juillet": 7,
        "août": 8, "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12, "decembre": 12}
NIVEAU = {"vert": 0, "orange": 1, "rouge": 2, "noir": 3}
ZONES = re.compile(r"^[1-6](?:-[1-6])*$")


def en_rgb(c):
    if c is None:
        return None
    if isinstance(c, (int, float)):
        return (float(c),) * 3
    c = tuple(float(v) for v in c)
    if len(c) == 1:
        return c * 3
    if len(c) == 3:
        return c
    if len(c) == 4:
        k = c[3]
        return tuple((1 - v) * (1 - k) for v in c[:3])
    return None


def classer(c):
    """Nom de la couleur Bison Futé, ou None (blanc, gris, violet des zones scolaires...).
    Classement par teinte, robuste aux variations RGB / CMJN d'une année à l'autre."""
    import colorsys
    rgb = en_rgb(c)
    if rgb is None:
        return None
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    h *= 360
    if v < 0.25:
        return "noir"
    if s < 0.35:
        return None
    if h < 14 or h >= 340:
        return "rouge"
    if h < 50:
        return "orange"
    if 65 <= h < 175:  # vert franc (2022+) ou vert-jaune (2017-2021)
        return "vert"
    return None


class Page:
    """Mots positionnés (pdfplumber) + image rendue de la page pour lire les couleurs.
    On lit les pixels plutôt que les rectangles : selon les années, les aplats sont en RGB, CMJN ou
    en tons directs (noir), que seul le rendu interprète correctement."""
    ECHELLE = 3.0

    def __init__(self, page):
        self.mots = page.extract_words(keep_blank_chars=False, use_text_flow=False)
        img = page.to_image(resolution=72 * self.ECHELLE).original.convert("RGB")
        self.px = np.asarray(img, dtype=float) / 255

    def couleur(self, x, y, rayon=1.0):
        """Couleur majoritaire d'un petit carré autour de (x, y), en points PDF."""
        h, w, _ = self.px.shape
        noms = []
        for dx in (-rayon, 0, rayon):
            for dy in (-rayon, 0, rayon):
                i = int(round((y + dy) * self.ECHELLE))
                j = int(round((x + dx) * self.ECHELLE))
                if 0 <= i < h and 0 <= j < w:
                    noms.append(classer(tuple(self.px[i, j])))
        noms = [n for n in noms if n]
        return max(set(noms), key=noms.count) if noms else None


def mois_de_la_page(pg: Page):
    """Repère chaque en-tête « Mois AAAA » et les mots Départ / Retour en dessous."""
    mots = pg.mots
    sortie = []
    for i, w in enumerate(mots):
        m = MOIS.get(w["text"].lower())
        if not m or i + 1 >= len(mots) or not re.fullmatch(r"20\d\d", mots[i + 1]["text"]):
            continue
        an = int(mots[i + 1]["text"])
        xc = (w["x0"] + mots[i + 1]["x1"]) / 2
        sous = [v for v in mots if 0 < v["top"] - w["top"] < 45 and abs((v["x0"] + v["x1"]) / 2 - xc) < 80]
        dep = [v for v in sous if v["text"].lower().startswith("d") and "part" in v["text"].lower()]
        ret = [v for v in sous if v["text"].lower().startswith("retour")]
        if not dep or not ret:
            continue
        # le plus proche de l'en-tête : les mois voisins sont parfois à moins de 80 points
        dep = min(dep, key=lambda v: abs((v["x0"] + v["x1"]) / 2 - xc))
        ret = [v for v in ret if v["x0"] > dep["x0"]]
        if not ret:
            continue
        ret = min(ret, key=lambda v: v["x0"] - dep["x0"])
        sortie.append(dict(an=an, mois=m, haut=max(dep["bottom"], ret["bottom"]),
                           cx_dep=(dep["x0"] + dep["x1"]) / 2, cx_ret=(ret["x0"] + ret["x1"]) / 2))
    return sortie


def lire_mois(pg: Page, mo):
    ecart = mo["cx_ret"] - mo["cx_dep"]
    bord = 0.37 * ecart  # distance du centre de colonne au bord de cellule
    # numéros de jours : à gauche de la colonne Départ, sous l'en-tête
    # le numéro est parfois collé à l'initiale du jour (« 10M ») selon les années
    jours = [w for w in pg.mots if re.fullmatch(r"\d{1,2}[LMJVSD]?", w["text"]) and w["top"] > mo["haut"]
             and mo["cx_dep"] - 1.6 * ecart < (w["x0"] + w["x1"]) / 2 < mo["cx_dep"] - 0.6 * ecart]
    vus, lignes = set(), []
    for j in sorted(jours, key=lambda w: w["top"]):
        jour = int(re.match(r"\d+", j["text"]).group())
        if jour in vus or jour > 31:
            continue  # un mois suivant plus bas dans la même colonne n'est pas géré ici
        try:
            date = pd.Timestamp(mo["an"], mo["mois"], jour)
        except ValueError:
            continue
        vus.add(jour)
        y = (j["top"] + j["bottom"]) / 2
        for sens, cx in (("depart", mo["cx_dep"]), ("retour", mo["cx_ret"])):
            nat = pg.couleur(cx - bord, y) or pg.couleur(cx + bord, y)
            if nat is None:
                continue
            if nat != "vert":
                lignes.append((date, sens, 0, nat))
            for w in pg.mots:
                wx, wy = (w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2
                if abs(wy - y) < 6 and abs(wx - cx) < bord and ZONES.fullmatch(w["text"]):
                    # au-dessus et au-dessous du texte (blanc), dans le cartouche
                    c = pg.couleur(wx, w["top"] - 2) or pg.couleur(wx, w["bottom"] + 2)
                    if c and NIVEAU[c] > NIVEAU[nat]:
                        lignes += [(date, sens, int(z), c) for z in w["text"].split("-")]
    return lignes, len(vus)


def extraire_pdf(chemin):
    import pdfplumber
    lignes, bilan = [], []
    with pdfplumber.open(chemin) as pdf:
        for page in pdf.pages:
            pg = Page(page)
            for mo in mois_de_la_page(pg):
                l, n = lire_mois(pg, mo)
                lignes += l
                bilan.append((mo["an"], mo["mois"], n))
    df = pd.DataFrame(lignes, columns=["date", "sens", "zone", "couleur"])
    return df, bilan


def lire_csv_2016(chemin):
    """Format openeventdatabase : date JJ/MM/AA ; aller, retour = suites de codes [zone]O|R|N."""
    c = pd.read_csv(chemin, dtype=str).fillna("")
    code = {"O": "orange", "R": "rouge", "N": "noir"}
    lignes = []
    for r in c.itertuples():
        date = pd.to_datetime(r.date, format="%d/%m/%y")
        for sens, cell in (("depart", r.aller), ("retour", r.retour)):
            for zone, lettre in re.findall(r"([1-6]?)([ORN])", cell):
                lignes.append((date, sens, int(zone) if zone else 0, code[lettre]))
    return pd.DataFrame(lignes, columns=["date", "sens", "zone", "couleur"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="+")
    ap.add_argument("--csv-2016")
    ap.add_argument("--sortie", default="bison_fute.csv")
    a = ap.parse_args()
    parts, apercus, annees = [], [], set()
    for chemin in a.pdf:
        df, bilan = extraire_pdf(chemin)
        an = pd.Series([b[0] for b in bilan]).mode()[0]  # année du calendrier
        incomplets = [f"{y}-{m:02d}:{n}" for y, m, n in bilan if n < 28 and y == an]
        print(f"{chemin} : {an}, {len(bilan)} mois, {len(df)} lignes non vertes"
              + (f", mois incomplets {incomplets}" if incomplets else ""))
        # chaque PDF déborde sur janvier suivant : cet aperçu ne sert que si l'année n'a pas son PDF
        parts.append(df[df["date"].dt.year == an])
        apercus.append(df[df["date"].dt.year != an])
        annees.add(an)
    if a.csv_2016:
        parts.append(lire_csv_2016(a.csv_2016))
        annees.add(2016)
    parts += [d[~d["date"].dt.year.isin(annees)] for d in apercus]
    out = pd.concat(parts).drop_duplicates(subset=["date", "sens", "zone"]).sort_values(["date", "sens", "zone"])
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(a.sortie, index=False)
    print(f"{len(out)} lignes -> {a.sortie}")


if __name__ == "__main__":
    main()
