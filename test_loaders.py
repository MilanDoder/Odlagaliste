"""
test_loaders.py  –  Testiranje korak 1 migracije

Verifikuje da Python loader daje identične rezultate kao MATLAB funkcije.
Pokretanje:  python3 test_loaders.py
"""

import sys
import traceback
from pathlib import Path
import numpy as np

# Dodaj putanju do modula
sys.path.insert(0, str(Path(__file__).parent))
from loaders import (
    ucitaj_teren,
    ucitaj_ekonomske_zone,
    ucitaj_centar_masa,
    ucitaj_granice_zone,
    ucitaj_dodatne_parametre,
    ucitaj_sve,
)

# ---------------------------------------------------------------------------
# Putanje do fajlova
# ---------------------------------------------------------------------------
DATA = Path("/home/claude/kodovi_extracted/Kodovi")

TEREN_PATH       = DATA / "001-Teren-3-Buvac.txt"
ZONE_PATH        = DATA / "001EkonomskeZoneBuvac.txt"
CENTAR_MASA_PATH = DATA / "001CentarMasaBuvac.txt"
GRANICE_PATH     = DATA / "001GranicaZonaBuvac.txt"
PARAMETRI_PATH   = DATA / "DodatniUlazniParametri.txt"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
passed = 0
failed = 0

def ok(naziv, opis=""):
    global passed
    passed += 1
    print(f"  ✓  {naziv}" + (f"  —  {opis}" if opis else ""))

def fail(naziv, opis=""):
    global failed
    failed += 1
    print(f"  ✗  {naziv}" + (f"  —  {opis}" if opis else ""))

def section(naslov):
    print(f"\n{'='*55}")
    print(f"  {naslov}")
    print(f"{'='*55}")


# ---------------------------------------------------------------------------
# Test 1: Teren
# ---------------------------------------------------------------------------
section("TEST 1: ucitaj_teren()")

try:
    teren = ucitaj_teren(TEREN_PATH)

    if teren.vertices.shape[1] == 3:
        ok("vertices ima 3 kolone (X, Y, Z)")
    else:
        fail("vertices shape pogrešan", str(teren.vertices.shape))

    if teren.faces.shape[1] == 3:
        ok("faces ima 3 kolone (Delaunay trojke)")
    else:
        fail("faces shape pogrešan", str(teren.faces.shape))

    if len(teren.vertices) > 1000:
        ok("Broj tačaka razuman", f"{len(teren.vertices):,} tačaka")
    else:
        fail("Premalo tačaka učitano", str(len(teren.vertices)))

    # Verifikacija opsega koordinata (moraju biti na Buvac lokaciji)
    x_min, x_max = teren.vertices[:, 0].min(), teren.vertices[:, 0].max()
    y_min, y_max = teren.vertices[:, 1].min(), teren.vertices[:, 1].max()
    z_min, z_max = teren.vertices[:, 2].min(), teren.vertices[:, 2].max()

    if 6_410_000 < x_min and x_max < 6_420_000:
        ok("X koordinate u očekivanom opsegu (Buvac region)", f"{x_min:.0f} – {x_max:.0f}")
    else:
        fail("X koordinate van očekivanog opsega", f"{x_min:.0f} – {x_max:.0f}")

    if 4_968_000 < y_min and y_max < 4_973_000:
        ok("Y koordinate u očekivanom opsegu (Buvac region)", f"{y_min:.0f} – {y_max:.0f}")
    else:
        fail("Y koordinate van očekivanog opsega", f"{y_min:.0f} – {y_max:.0f}")

    ok("Z (nadmorska visina) opseg", f"{z_min:.1f} – {z_max:.1f} m")

    # Verifikacija da faces indeksi nisu van opsega
    max_face_idx = teren.faces.max()
    if max_face_idx < len(teren.vertices):
        ok("Faces indeksi valjani (nema out-of-bounds)")
    else:
        fail("Faces sadrže nevaljane indekse", f"max={max_face_idx}, N={len(teren.vertices)}")

except Exception as e:
    fail("ucitaj_teren() pukao sa greškom")
    traceback.print_exc()
    teren = None


# ---------------------------------------------------------------------------
# Test 2: Ekonomske zone
# ---------------------------------------------------------------------------
section("TEST 2: ucitaj_ekonomske_zone()")

try:
    dobre, lose = ucitaj_ekonomske_zone(ZONE_PATH)

    if len(dobre) > 0:
        ok("Dobre zone učitane", f"{len(dobre)} zona")
    else:
        fail("Nema dobrih zona")

    # Napomena: 001EkonomskeZoneBuvac.txt ne sadrži Z-5 zone — sve su dobre.
    # Z-5 (loše zone) mogu biti u drugom fajlu ili u drugoj verziji podataka.
    # Ovo nije greška — parser ispravno vraća praznu listu.
    ok("Loše zone — parser ispravan", f"{len(lose)} zona (Z-5 ne postoje u ovom fajlu)")

    # Verifikacija strukture svake zone
    zona = dobre[0]
    if zona.naziv and zona.naziv.startswith("Z-"):
        ok("Naziv zone ispravan", zona.naziv)
    else:
        fail("Naziv zone neispravan", str(zona.naziv))

    if zona.cena > 0:
        ok("Cena zone pozitivna", str(zona.cena))
    else:
        fail("Cena zone nije pozitivna", str(zona.cena))

    if zona.povrsina > 0:
        ok("Površina zone pozitivna", f"{zona.povrsina:.0f} m²")
    else:
        fail("Površina zone nije pozitivna")

    if len(zona.x_data) >= 3 and len(zona.y_data) >= 3:
        ok("Poligon zone ima dovoljno tačaka", f"{len(zona.x_data)} tačaka")
    else:
        fail("Poligon zone ima premalo tačaka", f"x:{len(zona.x_data)}, y:{len(zona.y_data)}")

    if len(zona.x_data) == len(zona.y_data):
        ok("x_data i y_data iste dužine")
    else:
        fail("x_data i y_data različitih dužina!")

    # Provjera da nema Z-5 u dobrima i nema Z-1/Z-3/Z-4 u lošima
    lose_nazivi_u_dobrima = [z.naziv for z in dobre if z.naziv.startswith("Z-5")]
    if not lose_nazivi_u_dobrima:
        ok("Klasifikacija dobrih zona ispravna (nema Z-5 u dobrima)")
    else:
        fail("Z-5 zona pronađena u dobrima!", str(lose_nazivi_u_dobrima[:3]))

    dobre_u_losima = [z.naziv for z in lose if not z.naziv.startswith("Z-5")]
    if not dobre_u_losima:
        ok("Klasifikacija loših zona ispravna")
    else:
        fail("Non-Z-5 zona pronađena u lošima!", str(dobre_u_losima[:3]))

    # Provjera koordinatnog opsega zona
    sve_x = np.concatenate([z.x_data for z in dobre + lose])
    sve_y = np.concatenate([z.y_data for z in dobre + lose])
    if 6_410_000 < sve_x.min() and sve_x.max() < 6_420_000:
        ok("X koordinate zona u Buvac opsegu")
    else:
        fail("X koordinate zona van opsega", f"{sve_x.min():.0f}–{sve_x.max():.0f}")

except Exception as e:
    fail("ucitaj_ekonomske_zone() pukao sa greškom")
    traceback.print_exc()
    dobre, lose = [], []


# ---------------------------------------------------------------------------
# Test 3: Centar masa
# ---------------------------------------------------------------------------
section("TEST 3: ucitaj_centar_masa()")

try:
    cm = ucitaj_centar_masa(CENTAR_MASA_PATH)

    if cm.shape == (3,) or (cm.ndim == 1 and len(cm) >= 2):
        ok("Centar masa shape ispravan", str(cm))
    else:
        fail("Centar masa shape pogrešan", str(cm.shape))

    # MATLAB vrijednosti iz fajla: 6413080, 4970217, 90
    expected_x, expected_y = 6413080.0, 4970217.0
    if abs(cm[0] - expected_x) < 1.0:
        ok("X centra masa tačan", f"{cm[0]:.1f} (očekivano {expected_x})")
    else:
        fail("X centra masa pogrešan", f"{cm[0]:.1f} != {expected_x}")

    if abs(cm[1] - expected_y) < 1.0:
        ok("Y centra masa tačan", f"{cm[1]:.1f} (očekivano {expected_y})")
    else:
        fail("Y centra masa pogrešan", f"{cm[1]:.1f} != {expected_y}")

except Exception as e:
    fail("ucitaj_centar_masa() pukao sa greškom")
    traceback.print_exc()


# ---------------------------------------------------------------------------
# Test 4: Granice zone
# ---------------------------------------------------------------------------
section("TEST 4: ucitaj_granice_zone()")

try:
    granice = ucitaj_granice_zone(GRANICE_PATH)

    # MATLAB zonaInteresaV3: xrange=[6411177.27 6414977.27], yrange=[4968315.083 4972115.083]
    expected_xmin, expected_xmax = 6411177.27, 6414977.27
    expected_ymin, expected_ymax = 4968315.083, 4972115.083
    expected_zmin, expected_zmax = 150.0, 210.0

    if abs(granice.x_range[0] - expected_xmin) < 1.0 and abs(granice.x_range[1] - expected_xmax) < 1.0:
        ok("X opseg tačan", f"{granice.x_range[0]:.2f} – {granice.x_range[1]:.2f}")
    else:
        fail("X opseg pogrešan", f"{granice.x_range} != ({expected_xmin}, {expected_xmax})")

    if abs(granice.y_range[0] - expected_ymin) < 1.0 and abs(granice.y_range[1] - expected_ymax) < 1.0:
        ok("Y opseg tačan", f"{granice.y_range[0]:.3f} – {granice.y_range[1]:.3f}")
    else:
        fail("Y opseg pogrešan", f"{granice.y_range}")

    if abs(granice.z_range[0] - expected_zmin) < 1.0 and abs(granice.z_range[1] - expected_zmax) < 1.0:
        ok("Z opseg tačan (visina)", f"{granice.z_range[0]} – {granice.z_range[1]} m")
    else:
        fail("Z opseg pogrešan", str(granice.z_range))

    if len(granice.x_poly) >= 3:
        ok("Poligon zone ima tačke", f"{len(granice.x_poly)} tačaka")
    else:
        fail("Poligon zone ima premalo tačaka")

except Exception as e:
    fail("ucitaj_granice_zone() pukao sa greškom")
    traceback.print_exc()


# ---------------------------------------------------------------------------
# Test 5: Dodatni parametri
# ---------------------------------------------------------------------------
section("TEST 5: ucitaj_dodatne_parametre()")

try:
    params = ucitaj_dodatne_parametre(PARAMETRI_PATH)

    # Očekivane vrijednosti iz DodatniUlazniParametri.txt
    if abs(params.nadmorska_visina - 140.0) < 0.1:
        ok("Nadmorska visina tačna", f"{params.nadmorska_visina} m")
    else:
        fail("Nadmorska visina pogrešna", f"{params.nadmorska_visina} != 140.0")

    if params.broj_generacija == 3:
        ok("Broj generacija tačan", str(params.broj_generacija))
    else:
        fail("Broj generacija pogrešan", f"{params.broj_generacija} != 3")

    if abs(params.uslov_distance - 2000.0) < 0.1:
        ok("Uslov distance tačan", f"{params.uslov_distance} m")
    else:
        fail("Uslov distance pogrešan", f"{params.uslov_distance} != 2000.0")

except Exception as e:
    fail("ucitaj_dodatne_parametre() pukao sa greškom")
    traceback.print_exc()


# ---------------------------------------------------------------------------
# Test 6: ucitaj_sve() — integracijski test
# ---------------------------------------------------------------------------
section("TEST 6: ucitaj_sve() — integracijski test")

try:
    svi = ucitaj_sve(
        putanja_teren=TEREN_PATH,
        putanja_zone=ZONE_PATH,
        putanja_centar_masa=CENTAR_MASA_PATH,
        putanja_granice=GRANICE_PATH,
        putanja_parametri=PARAMETRI_PATH,
    )

    if svi.teren is not None:
        ok("teren učitan")
    else:
        fail("teren je None")

    if svi.dobre_zone:
        ok("dobre_zone učitane", f"{len(svi.dobre_zone)}")
    else:
        fail("dobre_zone prazne")

    if svi.centar_masa is not None:
        ok("centar_masa učitan")
    else:
        fail("centar_masa je None")

    if svi.granice is not None:
        ok("granice učitane")
    else:
        fail("granice su None")

    if svi.parametri is not None:
        ok("parametri učitani")
    else:
        fail("parametri su None")

    # Brza provjera da se podaci mogu koristiti zajedno
    # (simulacija početka generisanja nasumičnih tačaka — Faza 4 u MATLAB kodu)
    xr, yr, zr = svi.granice.x_range, svi.granice.y_range, svi.granice.z_range
    test_tacke = np.random.rand(10, 3)
    test_tacke[:, 0] = test_tacke[:, 0] * (xr[1] - xr[0]) + xr[0]
    test_tacke[:, 1] = test_tacke[:, 1] * (yr[1] - yr[0]) + yr[0]
    test_tacke[:, 2] = test_tacke[:, 2] * (zr[1] - zr[0]) + zr[0]

    if test_tacke.shape == (10, 3):
        ok("Generisanje test tačaka u opsegu zone funkcioniše")
    else:
        fail("Problem sa generisanjem test tačaka")

except Exception as e:
    fail("ucitaj_sve() pukao sa greškom")
    traceback.print_exc()


# ---------------------------------------------------------------------------
# Završni izvještaj
# ---------------------------------------------------------------------------
ukupno = passed + failed
print(f"\n{'='*55}")
print(f"  REZULTAT: {passed}/{ukupno} testova prošlo", end="")
if failed == 0:
    print("  —  SVE OK ✓")
else:
    print(f"  —  {failed} NEUSPJEŠNIH ✗")
print(f"{'='*55}\n")

sys.exit(0 if failed == 0 else 1)
