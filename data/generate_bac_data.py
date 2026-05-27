"""
Generator de date BAC Romania - date realiste bazate pe statistici reale
"""
import numpy as np
import pandas as pd
import random
import os

random.seed(42)
np.random.seed(42)

JUDETE = [
    "Alba", "Arad", "Arges", "Bacau", "Bihor", "Bistrita-Nasaud", "Botosani",
    "Brasov", "Braila", "Buzau", "Calarasi", "Cluj", "Constanta", "Covasna",
    "Dambovita", "Dolj", "Galati", "Giurgiu", "Gorj", "Harghita", "Hunedoara",
    "Ialomita", "Iasi", "Ilfov", "Maramures", "Mehedinti", "Mures", "Neamt",
    "Olt", "Prahova", "Satu Mare", "Salaj", "Sibiu", "Suceava", "Teleorman",
    "Timis", "Tulcea", "Vaslui", "Valcea", "Vrancea", "Bucuresti"
]

RATA_PROMOVARE_JUDET = {
    "Cluj": 0.78, "Timis": 0.76, "Sibiu": 0.75, "Brasov": 0.74, "Ilfov": 0.73,
    "Bucuresti": 0.72, "Arad": 0.71, "Bihor": 0.70, "Alba": 0.70, "Constanta": 0.69,
    "Hunedoara": 0.68, "Mures": 0.67, "Prahova": 0.67, "Iasi": 0.66, "Neamt": 0.65,
    "Arges": 0.65, "Bacau": 0.64, "Suceava": 0.64, "Maramures": 0.63, "Galati": 0.63,
    "Covasna": 0.63, "Harghita": 0.62, "Satu Mare": 0.62, "Dambovita": 0.61,
    "Vrancea": 0.61, "Dolj": 0.61, "Gorj": 0.60, "Valcea": 0.60, "Buzau": 0.60,
    "Salaj": 0.59, "Bistrita-Nasaud": 0.59, "Tulcea": 0.59, "Braila": 0.58,
    "Mehedinti": 0.58, "Olt": 0.57, "Botosani": 0.57, "Calarasi": 0.56,
    "Giurgiu": 0.55, "Ialomita": 0.55, "Teleorman": 0.53, "Vaslui": 0.50
}

SCOLI_PREFIXE = [
    "Colegiul National", "Liceul Teoretic", "Liceul Tehnologic",
    "Colegiul Tehnic", "Liceul Pedagogic", "Liceul de Arte",
    "Colegiul Economic", "Liceul Sportiv", "Scoala Nationala"
]

SCOLI_SUFIXE = [
    "Mihai Eminescu", "Nicolae Balcescu", "George Cosbuc", "Vasile Alecsandri",
    "Ioan Slavici", "Avram Iancu", "Stefan cel Mare", "Alexandru Ioan Cuza",
    "Gheorghe Sincai", "Octavian Goga", "Lucian Blaga", "Marin Preda",
    "Tudor Vladimirescu", "Dimitrie Cantemir", "Constantin Brancusi",
    "Mihai Viteazul", "Ion Luca Caragiale", "Traian", "Unirea",
    "Decebal", "Mircea cel Batran", "Bogdan Voda", "Elena Cuza",
    "Spiru Haret", "Onisifor Ghibu", "Emil Racovita", "Ion Barbu",
    "Grigore Moisil", "Ana Ipatescu", "Regina Maria"
]


def genereaza_nota(medie, std, minim=1.0, maxim=10.0):
    nota = np.random.normal(medie, std)
    nota = np.clip(nota, minim, maxim)
    return round(nota, 2)


def genereaza_date_bac(n_elevi=10000):
    date = []

    pentru_an = {
        2019: 1800, 2020: 1600, 2021: 1700, 2022: 1900, 2023: 1700, 2024: 1300
    }

    id_start = 1

    for an, n_an in pentru_an.items():
        for _ in range(n_an):
            judet = random.choice(JUDETE)
            rata_baza = RATA_PROMOVARE_JUDET[judet]

            mediu = "Urban" if random.random() < 0.60 else "Rural"
            if mediu == "Urban":
                rata_mediu = rata_baza * 1.12
            else:
                rata_mediu = rata_baza * 0.82

            gen = "F" if random.random() < 0.53 else "M"
            if gen == "F":
                rata_gen = rata_mediu * 1.05
            else:
                rata_gen = rata_mediu * 0.95

            rata_gen = min(rata_gen, 0.95)

            absent = 1 if random.random() < 0.05 else 0

            prefix = random.choice(SCOLI_PREFIXE)
            sufix = random.choice(SCOLI_SUFIXE)
            scoala = f"{prefix} {sufix}"

            if mediu == "Urban":
                nota_romana_oral = genereaza_nota(7.6, 1.4)
                nota_romana_scris = genereaza_nota(7.0, 1.8)
                nota_matematica = genereaza_nota(6.2, 2.1)
                nota_limba_straina = genereaza_nota(7.2, 1.7)
                nota_specialitate = genereaza_nota(7.0, 1.8)
            else:
                nota_romana_oral = genereaza_nota(7.1, 1.6)
                nota_romana_scris = genereaza_nota(6.4, 2.0)
                nota_matematica = genereaza_nota(5.6, 2.2)
                nota_limba_straina = genereaza_nota(6.5, 1.9)
                nota_specialitate = genereaza_nota(6.4, 2.0)

            # Ajustare bazata pe sansa de promovare
            if random.random() < rata_gen and not absent:
                # Elev cu sanse mari - note mai mari
                nota_romana_oral = max(nota_romana_oral, genereaza_nota(7.8, 1.2))
                nota_romana_scris = max(nota_romana_scris, genereaza_nota(7.2, 1.5))
                nota_matematica = max(nota_matematica, genereaza_nota(6.5, 1.8))
                nota_limba_straina = max(nota_limba_straina, genereaza_nota(7.4, 1.5))
                nota_specialitate = max(nota_specialitate, genereaza_nota(7.1, 1.6))

                # Asiguram cel putin 5 la toate materiile
                note = [nota_romana_oral, nota_romana_scris, nota_matematica,
                        nota_limba_straina, nota_specialitate]
                note = [max(n, 5.0) for n in note]
                nota_romana_oral, nota_romana_scris, nota_matematica, nota_limba_straina, nota_specialitate = note
            else:
                # Elev cu sanse mai mici - posibil note sub 5
                if random.random() < 0.3:
                    materie_slaba = random.randint(0, 4)
                    note = [nota_romana_oral, nota_romana_scris, nota_matematica,
                            nota_limba_straina, nota_specialitate]
                    note[materie_slaba] = genereaza_nota(3.5, 1.5, 1.0, 4.9)
                    nota_romana_oral, nota_romana_scris, nota_matematica, nota_limba_straina, nota_specialitate = note

            nota_romana_oral = round(min(max(nota_romana_oral, 1.0), 10.0), 2)
            nota_romana_scris = round(min(max(nota_romana_scris, 1.0), 10.0), 2)
            nota_matematica = round(min(max(nota_matematica, 1.0), 10.0), 2)
            nota_limba_straina = round(min(max(nota_limba_straina, 1.0), 10.0), 2)
            nota_specialitate = round(min(max(nota_specialitate, 1.0), 10.0), 2)

            medie_generala = round(
                (nota_romana_oral * 0.1 + nota_romana_scris * 0.3 +
                 nota_matematica * 0.3 + nota_limba_straina * 0.15 +
                 nota_specialitate * 0.15),
                2
            )

            if absent:
                promovat = 0
            elif (medie_generala >= 5.0 and
                  nota_romana_oral >= 5.0 and nota_romana_scris >= 5.0 and
                  nota_matematica >= 5.0 and nota_limba_straina >= 5.0 and
                  nota_specialitate >= 5.0):
                promovat = 1
            else:
                promovat = 0

            promotie = an

            date.append({
                "id": id_start,
                "an": an,
                "judet": judet,
                "scoala": scoala,
                "mediu": mediu,
                "gen": gen,
                "nota_romana_oral": nota_romana_oral,
                "nota_romana_scris": nota_romana_scris,
                "nota_matematica": nota_matematica,
                "nota_limba_straina": nota_limba_straina,
                "nota_specialitate": nota_specialitate,
                "medie_generala": medie_generala,
                "promovat": promovat,
                "absent": absent,
                "promotie": promotie
            })
            id_start += 1

    df = pd.DataFrame(date)
    return df


def main():
    print("Generare date BAC Romania...")
    df = genereaza_date_bac(10000)

    os.makedirs("data", exist_ok=True)
    output_path = os.path.join(os.path.dirname(__file__), "bac_date_romania.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Date generate: {len(df)} randuri")
    print(f"Rata promovare generala: {df['promovat'].mean():.1%}")
    print(f"Distributie gen: {df['gen'].value_counts().to_dict()}")
    print(f"Distributie mediu: {df['mediu'].value_counts().to_dict()}")
    print(f"Ani: {sorted(df['an'].unique())}")
    print(f"\nNota medie per materie:")
    for col in ['nota_romana_oral', 'nota_romana_scris', 'nota_matematica',
                'nota_limba_straina', 'nota_specialitate', 'medie_generala']:
        print(f"  {col}: {df[col].mean():.2f} (std: {df[col].std():.2f})")
    print(f"\nFisier salvat la: {output_path}")
    return df


if __name__ == "__main__":
    main()
