"""
Procesor date reale BAC Romania - data.gov.ro (2019-2024)
Suporta doua formate: standard (2019-2021, 2023-2024) si extins (2022 S1).
"""
import os
import pandas as pd
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "bac_date_romania.csv")

# Coduri SIIIR verificate din 2022 S1 (cross-referinta cu abrevieri oficiale)
JUDETE_COD = {
    "01": "Alba",           "02": "Arad",           "03": "Arges",
    "04": "Bacau",          "05": "Bihor",          "06": "Bistrita-Nasaud",
    "07": "Botosani",       "08": "Brasov",         "09": "Braila",
    "10": "Buzau",          "11": "Caras-Severin",  "12": "Cluj",
    "13": "Constanta",      "14": "Covasna",        "15": "Dambovita",
    "16": "Dolj",           "17": "Galati",         "18": "Gorj",
    "19": "Harghita",       "20": "Hunedoara",      "21": "Ialomita",
    "22": "Iasi",           "23": "Ilfov",          "24": "Maramures",
    "25": "Mehedinti",      "26": "Mures",          "27": "Neamt",
    "28": "Olt",            "29": "Prahova",        "30": "Satu Mare",
    "31": "Salaj",          "32": "Sibiu",          "33": "Suceava",
    "34": "Teleorman",      "35": "Timis",          "36": "Tulcea",
    "37": "Vaslui",         "38": "Valcea",         "39": "Vrancea",
    "40": "Bucuresti",      "51": "Calarasi",       "52": "Giurgiu",
}

# Abrevieri oficiale judete (format 2022 S1)
JUDETE_ABREV = {
    "AB": "Alba",           "AR": "Arad",           "AG": "Arges",
    "BC": "Bacau",          "BH": "Bihor",          "BN": "Bistrita-Nasaud",
    "BT": "Botosani",       "BV": "Brasov",         "BR": "Braila",
    "BZ": "Buzau",          "CS": "Caras-Severin",  "CJ": "Cluj",
    "CT": "Constanta",      "CV": "Covasna",        "DB": "Dambovita",
    "DJ": "Dolj",           "GL": "Galati",         "GJ": "Gorj",
    "HR": "Harghita",       "HD": "Hunedoara",      "IL": "Ialomita",
    "IS": "Iasi",           "IF": "Ilfov",          "MM": "Maramures",
    "MH": "Mehedinti",      "MS": "Mures",          "NT": "Neamt",
    "OT": "Olt",            "PH": "Prahova",        "SM": "Satu Mare",
    "SJ": "Salaj",          "SB": "Sibiu",          "SV": "Suceava",
    "TR": "Teleorman",      "TM": "Timis",          "TL": "Tulcea",
    "VS": "Vaslui",         "VL": "Valcea",         "VN": "Vrancea",
    "B":  "Bucuresti",      "CL": "Calarasi",       "GR": "Giurgiu",
}

FILES = [
    ("bac_2019_s2.xlsx", 2019, "xlsx"),
    ("bac_2020_s1.xlsx", 2020, "xlsx"),
    ("bac_2020_s2.xlsx", 2020, "xlsx"),
    ("bac_2021_s1.xlsx", 2021, "xlsx"),
    ("bac_2021_s2.xlsx", 2021, "xlsx"),
    ("bac_2022_s1.xlsx", 2022, "xlsx"),
    ("bac_2022_s2.xlsx", 2022, "xlsx"),
    ("bac_2023_s1.xlsx", 2023, "xlsx"),
    ("bac_2023_s2.xlsx", 2023, "xlsx"),
    ("bac_2024_s1.xlsx", 2024, "xlsx"),
    ("bac_2024_s2.xlsx", 2024, "xlsx"),
]


def judet_din_siiir(val) -> str:
    try:
        s = str(val).strip().split(".")[0].zfill(10)
        return JUDETE_COD.get(s[:2], "")
    except Exception:
        return ""


def nota_finala(df, nota_col, contestatie_col=None, nota_contest_col=None):
    def _get(row):
        try:
            nota = float(row[nota_col])
        except (ValueError, TypeError):
            return np.nan
        if pd.isna(nota) or nota < 0:
            return np.nan
        if contestatie_col and nota_contest_col:
            try:
                if str(row.get(contestatie_col, "")).strip().lower() == "da":
                    nc = float(row[nota_contest_col])
                    if not pd.isna(nc) and nc > nota:
                        return round(nc, 2)
            except (ValueError, TypeError):
                pass
        return round(nota, 2)
    return df.apply(_get, axis=1)


def status_oral(series):
    return series.apply(
        lambda x: {"promovat": 8.0, "nepromovat": 3.0}.get(
            str(x).strip().lower(), np.nan)
    )


def promotie_an(series, default_an):
    def _p(x):
        try:
            v = int(str(x).strip()[-4:])
            return v if 2000 <= v <= 2030 else default_an
        except Exception:
            return default_an
    return series.apply(_p)


def proceseaza_standard(df, an):
    """Format: 'Sex', 'Mediu candidat', 'Unitate (SIIIR)', 'STATUS', 'Medie'"""
    r = pd.DataFrame()
    r["judet"] = df["Unitate (SIIIR)"].apply(judet_din_siiir)
    r["scoala"] = "Unitate " + df["Unitate (SIIIR)"].astype(str).str.split(".").str[0].str.strip()
    r["gen"] = df["Sex"].str.strip().str.upper().map({"M": "M", "F": "F"})
    r["mediu"] = df["Mediu candidat"].str.strip().str.capitalize().map(
        {"Urban": "Urban", "Rural": "Rural"}
    )
    r["an"] = an
    prom = df["Promoție"] if "Promoție" in df.columns else pd.Series([an] * len(df))
    r["promotie"] = promotie_an(prom, an)
    r["nota_romana_oral"] = status_oral(df["STATUS_A"]) if "STATUS_A" in df.columns else np.nan
    r["nota_romana_scris"] = nota_finala(df, "NOTA_EA", "CONTESTATIE_EA", "NOTA_CONTESTATIE_EA")
    r["nota_limba_straina"] = (
        nota_finala(df, "NOTA_EB", "CONTESTATIE_EB", "NOTA_CONTESTATIE_EB")
        if "NOTA_EB" in df.columns else np.nan
    )
    r["nota_matematica"] = nota_finala(df, "NOTA_EC", "CONTESTATIE_EC", "NOTA_CONTESTATIE_EC")
    r["nota_specialitate"] = (
        nota_finala(df, "NOTA_ED", "CONTESTATIE_ED", "NOTA_CONTESTATIE_ED")
        if "NOTA_ED" in df.columns else np.nan
    )
    r["medie_generala"] = pd.to_numeric(df["Medie"], errors="coerce").round(2)
    status = df["STATUS"].str.strip().str.lower()
    r["promovat"] = (status == "promovat").astype(int)
    r["absent"] = (status == "absent").astype(int)
    return r


def proceseaza_extins(df, an):
    """Format 2022 S1: 'Judet' (abreviere), 'SCOALA', 'SEX', 'MEDIU', 'STATUS_FINAL', 'MEDIA_FINALA'"""
    r = pd.DataFrame()
    r["judet"] = df["Judet"].str.strip().str.upper().map(JUDETE_ABREV).fillna("")
    r["scoala"] = df["SCOALA"].str.strip() if "SCOALA" in df.columns else ""
    r["gen"] = df["SEX"].str.strip().str.lower().map(
        {"m": "M", "f": "F", "masculin": "M", "feminin": "F"}
    )
    r["mediu"] = df["MEDIU"].str.strip().str.capitalize().map(
        {"Urban": "Urban", "Rural": "Rural"}
    )
    r["an"] = an
    prom_col = next((c for c in ["PROMOTIA_CURENTA", "PROMOTIA"] if c in df.columns), None)
    r["promotie"] = promotie_an(df[prom_col], an) if prom_col else an
    r["nota_romana_oral"] = status_oral(df["STATUS_A"]) if "STATUS_A" in df.columns else np.nan
    r["nota_romana_scris"] = nota_finala(df, "NOTA_EA", "CONTESTATIE_EA", "NOTA_CONTESTATIE_EA")
    r["nota_limba_straina"] = (
        nota_finala(df, "NOTA_EB", "CONTESTATIE_EB", "NOTA_CONTESTATIE_EB")
        if "NOTA_EB" in df.columns else np.nan
    )
    r["nota_matematica"] = nota_finala(df, "NOTA_EC", "CONTESTATIE_EC", "NOTA_CONTESTATIE_EC")
    r["nota_specialitate"] = (
        nota_finala(df, "NOTA_ED", "CONTESTATIE_ED", "NOTA_CONTESTATIE_ED")
        if "NOTA_ED" in df.columns else np.nan
    )
    medie_col = "MEDIA_FINALA" if "MEDIA_FINALA" in df.columns else "Medie"
    r["medie_generala"] = pd.to_numeric(df[medie_col], errors="coerce").round(2)
    status = df["STATUS_FINAL"].str.strip().str.lower()
    r["promovat"] = (status == "promovat").astype(int)
    r["absent"] = (status == "absent").astype(int)
    return r


def detecteaza_format(df):
    return "extins" if ("SEX" in df.columns and "STATUS_FINAL" in df.columns) else "standard"


def proceseaza_fisier(path, an, fmt):
    print(f"  {os.path.basename(path)} (an={an})", end=" ... ", flush=True)
    try:
        df = pd.read_excel(path, dtype={"Unitate (SIIIR)": str, "Cod SIIIR": str})
    except Exception as e:
        print(f"EROARE: {e}")
        return None

    tip = detecteaza_format(df)
    try:
        r = proceseaza_extins(df, an) if tip == "extins" else proceseaza_standard(df, an)
    except Exception as e:
        print(f"EROARE procesare [{tip}]: {e}")
        return None

    r = r.dropna(subset=["gen", "mediu"])
    r = r[r["judet"].str.len() > 0]
    r = r.reset_index(drop=True)
    r.insert(0, "id", range(1, len(r) + 1))

    note_cols = ["nota_romana_oral", "nota_romana_scris", "nota_matematica",
                 "nota_limba_straina", "nota_specialitate", "medie_generala"]
    for col in note_cols:
        r[col] = pd.to_numeric(r[col], errors="coerce").clip(1.0, 10.0)

    print(f"{len(r):,} randuri [{tip}]")
    return r


def main():
    toate = []
    id_curent = 1

    for fname, an, fmt in FILES:
        path = os.path.join(RAW_DIR, fname)
        if not os.path.exists(path):
            print(f"  LIPSA: {fname}")
            continue
        df = proceseaza_fisier(path, an, fmt)
        if df is not None and not df.empty:
            df["id"] = range(id_curent, id_curent + len(df))
            id_curent += len(df)
            toate.append(df)

    if not toate:
        print("EROARE: Niciun fisier procesat!")
        return

    final = pd.concat(toate, ignore_index=True)
    final["id"] = range(1, len(final) + 1)
    final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    judete = sorted(final["judet"].unique().tolist())
    print(f"\n=== DATE REALE BAC ROMANIA ===")
    print(f"Total elevi: {len(final):,}")
    print(f"Ani: {sorted(final['an'].unique().tolist())}")
    print(f"Judete ({len(judete)}): {judete}")
    print(f"Rata promovare: {final['promovat'].mean():.1%}")
    print(f"Gen: {final['gen'].value_counts().to_dict()}")
    print(f"Mediu: {final['mediu'].value_counts().to_dict()}")
    note_cols = ["nota_romana_scris", "nota_matematica", "nota_limba_straina",
                 "nota_specialitate", "medie_generala"]
    print(f"\nNote medii:")
    for col in note_cols:
        m = final[col].mean()
        print(f"  {col}: {m:.2f}" if not pd.isna(m) else f"  {col}: N/A")
    print(f"\nSalvat: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
