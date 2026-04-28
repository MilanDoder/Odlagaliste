# Optimizacija lokacije odlagališta — Python migracija

Potpuna Python migracija MATLAB projekta `IzvrsniKodBuvac.m`.  
Genetski algoritam za optimizaciju lokacije rudarskog odlagališta.

---

## Struktura projekta

```
odlagaliste/
├── loaders.py          # Korak 1 — uvoz podataka
├── geometry.py         # Korak 2 — geometrija kupe (Möller, ConvexHull)
├── ekonomija.py        # Korak 3a — ekonomski proračun (11 zona)
├── ga_funkcije.py      # Korak 3b — funkcija cilja i ograničenja GA
├── ga_pokretac.py      # Korak 4 — GA pokretač (scipy differential_evolution)
├── izvoz.py            # Korak 5 — Excel (CSV) i DXF izvoz
├── main.py             # Glavni pokretač (CLI)
├── app.py              # Streamlit web sučelje
├── config.example.json # Primjer konfiguracije
└── podaci/             # Ovdje stavi ulazne fajlove
    ├── 001-Teren-3-Buvac.txt
    ├── 001EkonomskeZoneBuvac.txt
    ├── 001CentarMasaBuvac.txt
    ├── 001GranicaZonaBuvac.txt
    └── DodatniUlazniParametri.txt
```

---

## Instalacija

```bash
pip install numpy scipy matplotlib streamlit plotly pandas openpyxl
```

Minimalno (bez UI i Excel):
```bash
pip install numpy scipy matplotlib
```

---

## Pokretanje

### Opcija 1: Komandna linija (zamjena za MATLAB konzolu)

```bash
# Sa config fajlom
cp config.example.json config.json
# Uredi config.json sa tačnim putanjama
python3 main.py --config config.json

# Direktno sa argumentima
python3 main.py \
  --teren   podaci/001-Teren-3-Buvac.txt \
  --zone    podaci/001EkonomskeZoneBuvac.txt \
  --cm      podaci/001CentarMasaBuvac.txt \
  --granice podaci/001GranicaZonaBuvac.txt \
  --params  podaci/DodatniUlazniParametri.txt \
  --tacke 200 --ponavljanja 3 --ugao 37

# Interaktivno (kao MATLAB — pita za svaki parametar)
python3 main.py --interaktivno
```

### Opcija 2: Web sučelje (zamjena za MATLAB figure + input)

```bash
streamlit run app.py
```

Otvori browser na `http://localhost:8501`:
- Upload fajlove terena, zona, centra masa
- Podesi parametre sliderima
- Klikni **▶ Pokreni optimizaciju**
- Preuzmi rezultate (CSV + DXF u ZIP)

---

## Format ulaznih fajlova

### Teren (`001-Teren-3-Buvac.txt`)
CSV bez zaglavlja, tri kolone: X, Y, Z
```
6412550.704,4970877.846,156.470
6413476.252,4971541.634,160.641
...
```

### Ekonomske zone (`001EkonomskeZoneBuvac.txt`)
5 linija po zoni: naziv, cijena, površina, X koordinate, Y koordinate
```
Z-1-1.1
15
55737
6411777,6411577,6411577,...
4970315,4970315,4970115,...
```

### Centar masa (`001CentarMasaBuvac.txt`)
Jedna linija: X, Y, Z
```
6413080,4970217,90
```

### Granice zone (`001GranicaZonaBuvac.txt`)
Prva linija: X_min, Y_min, Z_min  
Druga linija: X_max, Y_max, Z_max  
Ostale linije: poligon interesne zone
```
6411177.27,4968315.083,150
6414977.27,4972115.083,210
6411177.27,4968315.083,150
...
```

### Dodatni parametri (`DodatniUlazniParametri.txt`)
```
%% mv - nadmorska visina
140
%% broj generacija
3
%% uslov distance transporta
2000
```

---

## GA varijable i bounds

| Varijabla | Opis | Buvac bounds | V1 bounds |
|---|---|---|---|
| `x[0]` = wz | Visina vrha kupe (m) | 175–280 | 155–210 |
| `x[1]` = k  | Širina kupe (m) | 80–350 | 70–120 |
| `x[2]` = wx | X koordinata (fiksna) | pointX | pointX |
| `x[3]` = wy | Y koordinata (fiksna) | pointY | pointY |

---

## Izlaz

Excel/CSV fajlovi sa 13 kolona (identično MATLAB `headerFinal`):

| Kolona | Opis |
|---|---|
| Naziv_tacke | Identifikator tačke |
| X_koordinata, Y_koordinata, Z_koordinata | Koordinate vrha |
| K | Optimalna širina kupe |
| Funkcija_cilja | Vrijednost f (minimizovana) |
| Zapremina | Zapremina kupe (m³) |
| Ugao | Ugao kosine (°) |
| distanca | Distanca od centra masa (m) |
| c1 | Transportni trošak |
| c2 | Trošak iskapanja |
| c3 | Vrijednost zemljišta |
| Zone | Ekonomske zone koje kupa pokriva |

DXF fajl sadrži gornju i donju konturu svake kupe, svaka na zasebnom sloju.

---

## MATLAB → Python mapiranje

| MATLAB | Python | Fajl |
|---|---|---|
| `uvozTerenaV3.m` + `uigetfile()` | `ucitaj_teren()` | `loaders.py` |
| `uvozEkonomskihZonaBuvac.m` | `ucitaj_ekonomske_zone()` | `loaders.py` |
| 39 getter/setter `.m` fajlova | `UlazniPodaci` dataclass | `loaders.py` |
| `SurfaceIntersection.m` (885 linija) | `surface_intersection()` — Möller | `geometry.py` |
| `pol2cart()`, `delaunay()`, `convhull()` | numpy/scipy ekvivalenti | `geometry.py` |
| `proracunEkonomskeCeneSaKoeficijentom.m` | `cijena_zone()` | `ekonomija.py` |
| `funkcijaCiljaGenetskiAlgoritam.m` | `funkcija_cilja()` | `ga_funkcije.py` |
| `ga()` — MATLAB Global Optimization Toolbox | `scipy.optimize.differential_evolution` | `ga_pokretac.py` |
| `writetable()` + `array2table()` | `izvezi_excel()` — pandas/CSV | `izvoz.py` |
| 10× `dxf_*.m` fajlova | `DxfWriter` klasa | `izvoz.py` |
| MATLAB figure + `trisurf()` | Plotly 3D interaktivni grafikoni | `app.py` |
| `input()` konzola | Streamlit sidebar | `app.py` |

---

## Testovi

```bash
python3 test_loaders.py       # 33/33
python3 test_geometry.py      # 16/16
python3 test_ekonomija_ga.py  # 22/22
python3 test_ga_izvoz.py      # 24/24
```

Ukupno: **95/95 testova**
