# BAC Romania — Statistici & Predictii ML

Aplicatie web pentru analiza si predictia rezultatelor la Bacalaureat in Romania (2019–2024), construita cu **Streamlit**, **scikit-learn** si **MongoDB**.

Datele provin exclusiv din seturile oficiale publicate pe [data.gov.ro](https://data.gov.ro) — **908.401 elevi reali**.

---

## Functionalitati

| Sectiune | Descriere |
|---|---|
| Dashboard | KPI-uri nationale: rata promovare, medie generala, elevi pe an |
| Evolutie Anuala | Grafic tendinte 2019–2024 pe sesiuni |
| Statistici pe Judet | Harta interactiva + clasament judete |
| Statistici pe Materie | Distributia notelor la Romana, Proba C, D, Limba Straina |
| Analiza Gen | Comparatie masculin / feminin |
| Urban vs Rural | Diferente de performanta mediu urban / rural |
| Top Scoli | Clasament scoli dupa rata promovare si medie |
| Distributie Note | Histograme si boxplot-uri |
| Predictie ML | Estimeaza probabilitatea de promovare pentru un elev nou |
| Live Dashboard | Simulare flux date in timp real |

---

## Structura proiect

```
Schoolpred/
├── app.py                        # Aplicatia Streamlit principala
├── requirements.txt              # Dependinte Python
├── .gitignore
├── data/
│   ├── process_real_data.py      # Procesor date oficiale BAC
│   ├── raw/                      # Fisiere XLSX/CSV originale (excluse din git)
│   └── bac_date_romania.csv      # CSV procesat (exclus din git, ~150 MB)
├── database/
│   └── mongodb_client.py         # Client MongoDB cu fallback CSV
├── models/
│   ├── predictor.py              # Model ML (RandomForest + GradientBoosting)
│   └── saved/                    # Modele antrenate salvate (excluse din git)
└── .streamlit/
    └── config.toml               # Tema Streamlit
```

---

## Instalare

### 1. Cerinte

- Python 3.10+
- MongoDB 6+ (optional — aplicatia functioneaza si fara el, cu fallback CSV)

### 2. Dependinte Python

```bash
pip install -r requirements.txt
pip install statsmodels   # necesar pentru grafice cu trendline
```

### 3. Date reale BAC

Ai doua optiuni:

#### Optiunea A — Importi direct CSV-ul procesat (recomandat)

Descarca fisierul `bac_date_romania.csv` si plaseaza-l in `data/`:

```
Schoolpred/
└── data/
    └── bac_date_romania.csv   ← aici
```

Aplica direct `python app.py` / `streamlit run app.py`. CSV-ul va fi incarcat automat.

#### Optiunea B — Procesezi datele brute de la zero

1. Descarca fisierele XLSX/CSV oficiale de pe [data.gov.ro](https://data.gov.ro) (cauta "bacalaureat"):

   | Fisier | An | Sesiune |
   |---|---|---|
   | `bac_2019_s2.xlsx` | 2019 | Sesiunea 2 |
   | `bac_2020_s1.xlsx` | 2020 | Sesiunea 1 |
   | `bac_2020_s2.xlsx` | 2020 | Sesiunea 2 |
   | `bac_2021_s1.xlsx` | 2021 | Sesiunea 1 |
   | `bac_2021_s2.xlsx` | 2021 | Sesiunea 2 |
   | `bac_2022_s1.xlsx` | 2022 | Sesiunea 1 (format diferit) |
   | `bac_2022_s2.xlsx` | 2022 | Sesiunea 2 |
   | `bac_2023_s1.xlsx` | 2023 | Sesiunea 1 |
   | `bac_2023_s2.xlsx` | 2023 | Sesiunea 2 |
   | `bac_2024_s1.xlsx` | 2024 | Sesiunea 1 |
   | `bac_2024_s2.xlsx` | 2024 | Sesiunea 2 |

2. Plaseaza-le in `data/raw/`

3. Ruleaza procesorul:

   ```bash
   python data/process_real_data.py
   ```

   Scriptul detecteaza automat formatul fiecarui fisier (standard 2019–2021 / extins 2022 S1), mapeaza codurile SIIIR la judete si genereaza `data/bac_date_romania.csv`.

---

## Rulare aplicatie

```bash
streamlit run app.py
```

Aplicatia se deschide automat la `http://localhost:8501`.

---

## Baza de date MongoDB (optional)

Aplicatia functioneaza si fara MongoDB — datele se incarca direct din CSV.

Daca vrei sa activezi MongoDB:

1. Porneste serviciul:
   ```bash
   sudo systemctl start mongod
   ```

2. La prima rulare, aplicatia detecteaza MongoDB activ si importa automat CSV-ul in colectia `elevi` din baza `bac_romania`.

3. La rulaile urmatoare datele vin din MongoDB (mai rapid).

Configurare implicita (schimba in `database/mongodb_client.py` daca e nevoie):
```
URI:  mongodb://localhost:27017/
DB:   bac_romania
```

---

## Modelul ML

Predictor dual antrenat pe toate cele 908.401 inregistrari:

- **RandomForestClassifier** — prezice daca elevul promoveaza (0/1)
- **GradientBoostingRegressor** — estimeaza media generala

Features folosite:
- Nota Romana Scris
- Proba C (Matematica sau alta disciplina)
- Limba Straina
- Proba D (Specialitate)
- Mediu (urban / rural)
- Gen (masculin / feminin)
- Anul sesiunii

Valorile lipsa (note necompletate) sunt imputate automat cu mediana.

Modelul se antreneaza automat la prima accesare a sectiunii **Predictie ML** si se salveaza in `models/saved/`.

---

## Date sursa

Toate datele provin din fisierele oficiale publicate de **Ministerul Educatiei** prin:

- [data.gov.ro — Bacalaureat](https://data.gov.ro) (cautare: "bacalaureat rezultate")

Perioadele acoperite: **2019 S2 — 2024 S1** | **908.401 elevi** | **42 judete**

---

## Schema CSV

Fisierul `bac_date_romania.csv` contine urmatoarele coloane:

| Coloana | Tip | Descriere |
|---|---|---|
| `an` | int | Anul sesiunii BAC |
| `sesiune` | str | S1 / S2 |
| `judet` | str | Numele judetului |
| `mediu` | str | `urban` / `rural` |
| `gen` | str | `masculin` / `feminin` |
| `nota_romana_scris` | float | Nota finala Romana scris (1–10) |
| `nota_matematica` | float | Nota Proba C |
| `nota_limba_straina` | float | Nota Limba Straina |
| `nota_specialitate` | float | Nota Proba D |
| `medie_generala` | float | Media generala calculata |
| `promovat` | int | 1 = promovat, 0 = nepromovat |
| `absent` | int | 1 = absent, 0 = prezent |
