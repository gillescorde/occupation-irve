# Calendriers Bison Futé bruts (récupérés le 17/10/2026)

Un PDF par année, nommé `bison_fute_AAAA.pdf`, téléchargé tel quel depuis bison-fute.gouv.fr.
Les couleurs y sont des aplats graphiques, pas du texte : l'extraction reste à faire.

| Année | Fichier | Source |
|---|---|---|
| 2016 | `bison_fute_2016.csv` (pas de PDF) | https://raw.githubusercontent.com/openeventdatabase/datasources/master/fr.bisonfute/bisonfute2016.csv (ressource b79344d0… du jeu data.gouv « Prévisions de trafic de bison futé ») |
| 2017 | `bison_fute_2017.pdf` (2 p., A3) | https://www.bison-fute.gouv.fr/IMG/pdf/Calendrier_2017.pdf |
| 2018 | `bison_fute_2018.pdf` (2 p., A3) | https://www.bison-fute.gouv.fr/IMG/pdf/17237_Calendrier_2018_Bison_Fute_A3_DEF_Web-2.pdf |
| 2019 | `bison_fute_2019.pdf` (1 p., A3) | https://www.bison-fute.gouv.fr/IMG/pdf/Calendrier_2019_Bison_Fute.pdf |
| 2021 | `bison_fute_2021.pdf` (3 p., A3) | https://www.bison-fute.gouv.fr/IMG/pdf/Calendrier_2021_BisonFute.pdf |
| 2022 | `bison_fute_2022.pdf` (2 p.) | https://www.bison-fute.gouv.fr/IMG/pdf/Calendrier_2022-Bison_Fute-Impression.pdf |
| 2023 | `bison_fute_2023.pdf` (3 p., A3) | https://www1.bison-fute.gouv.fr/IMG/pdf/Calendrier_2023_Bison_Fute_A3.pdf |
| 2024 | `bison_fute_2024.pdf` (2 p., A2) | https://www.bison-fute.gouv.fr/IMG/pdf/Calendrier_2024_Bison_Fute_-2.pdf |
| 2025 | `bison_fute_2025.pdf` (2 p., A2) | https://www.bison-fute.gouv.fr/IMG/pdf/Calendrier_Bison_fute_2025.pdf |
| 2026 | `bison_fute_2026.pdf` (2 p., A2) | https://www.bison-fute.gouv.fr/IMG/pdf/Calendrier_Bison_Fute_2026.pdf |

## Années manquantes

- **2016 (PDF)** : l'URL d'origine `15209_Calendrier_2016_Bison_Fute_A3_DEF_Light.pdf` renvoie 404 et aucune variante testée n'existe. On n'a que le CSV.
- **2020** : aucun PDF trouvé sur bison-fute.gouv.fr (noms testés : Calendrier_2020_Bison_Fute.pdf, Calendrier_2020_BisonFute.pdf, Calendrier_2020.pdf, etc.). Une copie existe sur Scribd (document 470280479), non téléchargeable ici.
- web.archive.org est bloqué depuis l'environnement cloud, d'où l'absence de récupération par archive. À télécharger à la main si besoin.

Des versions pour daltoniens existent pour 2021 (`Calendrier_2021_BisonFute_daltonien.pdf`) et 2026 (`Calendrier_Bison_Fute_2026_version_daltonien.pdf`) ; elles ne sont pas téléchargées.

## Format du CSV 2016

71 lignes, uniquement les jours non verts. Colonnes `date` (JJ/MM/AA), `aller` (départs), `retour`.
Chaque cellule enchaîne des codes couleur `O` (orange), `R` (rouge), `N` (noir), précédés ou non d'un chiffre de zone :
une lettre sans chiffre vaut pour toute la France, `4R` pour la zone 4 seule.
Exemple : `O4R` = orange sur toute la France, rouge en zone 4 ; `4O6O` = orange en zones 4 et 6. Cellule vide = vert.
