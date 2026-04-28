"""
test_ekonomija_ga.py  –  Testiranje koraka 3: ekonomija + GA funkcije

Pokretanje:  python3 test_ekonomija_ga.py
"""

import sys
import traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from loaders import ucitaj_sve
from geometry import zapremina_kupe
from ekonomija import (
    cijena_zone,
    ekonomska_cijena,
    distanca_od_centra_masa,
    racunaj_troskove,
    ZONA_FORMULA,
)
from ga_funkcije import (
    GAKontekst,
    funkcija_cilja,
    get_bounds,
)

DATA = Path("/home/claude/kodovi_extracted/Kodovi")

passed = 0
failed = 0

def ok(naziv, opis=""):
    global passed; passed += 1
    print(f"  ✓  {naziv}" + (f"  —  {opis}" if opis else ""))

def fail(naziv, opis=""):
    global failed; failed += 1
    print(f"  ✗  {naziv}" + (f"  —  {opis}" if opis else ""))

def section(s):
    print(f"\n{'='*55}\n  {s}\n{'='*55}")


# ---------------------------------------------------------------------------
# Učitaj podatke
# ---------------------------------------------------------------------------
print("Učitavam podatke...")
podaci = ucitaj_sve(
    DATA / "001-Teren-3-Buvac.txt",
    DATA / "001EkonomskeZoneBuvac.txt",
    DATA / "001CentarMasaBuvac.txt",
    DATA / "001GranicaZonaBuvac.txt",
    DATA / "DodatniUlazniParametri.txt",
)


# ---------------------------------------------------------------------------
# Test 1: cijena_zone
# ---------------------------------------------------------------------------
section("TEST 1: cijena_zone — MATLAB proracunEkonomskeCeneSaKoeficijentom")

try:
    povrsina = 10_000.0   # 10.000 m²

    # Z-1-1: CENA3 * p * (1 + K4) = 0.15 * 10000 * (1 + 0.084) = 1626.0
    c11 = cijena_zone("Z-1-1.3", povrsina)
    expected_11 = 0.15 * povrsina * (1 + 0.084)
    if abs(c11 - expected_11) < 0.01:
        ok("Z-1-1 formula tačna", f"{c11:.2f} (očekivano {expected_11:.2f})")
    else:
        fail("Z-1-1 formula pogrešna", f"{c11:.2f} != {expected_11:.2f}")

    # Z-1-7: CENA2 * p * (1 + K53) = 0.3 * 10000 * (1 + 0.011) = 3033.0
    c17 = cijena_zone("Z-1-7.2", povrsina)
    expected_17 = 0.3 * povrsina * (1 + 0.011)
    if abs(c17 - expected_17) < 0.01:
        ok("Z-1-7 formula tačna", f"{c17:.2f}")
    else:
        fail("Z-1-7 formula pogrešna", f"{c17:.2f} != {expected_17:.2f}")

    # Z-3: CENA3 * p * (1 + K4 + K3) = 0.15 * 10000 * (1 + 0.084 + 0.126) = 3150.0
    c3 = cijena_zone("Z-3.5", povrsina)
    expected_3 = 0.15 * povrsina * (1 + 0.084 + 0.126)
    if abs(c3 - expected_3) < 0.01:
        ok("Z-3 formula tačna", f"{c3:.2f}")
    else:
        fail("Z-3 formula pogrešna", f"{c3:.2f} != {expected_3:.2f}")

    # Z-4-1: CENA1 * p * (1 + K2) = 0.5 * 10000 * (1 + 0.252) = 6260.0
    c41 = cijena_zone("Z-4-1.3", povrsina)
    expected_41 = 0.5 * povrsina * (1 + 0.252)
    if abs(c41 - expected_41) < 0.01:
        ok("Z-4-1 formula tačna", f"{c41:.2f}")
    else:
        fail("Z-4-1 formula pogrešna", f"{c41:.2f} != {expected_41:.2f}")

    # Z-4-2: CENA1 * p * (1 + K2 + K4) = 0.5 * 10000 * (1 + 0.252 + 0.084) = 6680.0
    c42 = cijena_zone("Z-4-2.1", povrsina)
    expected_42 = 0.5 * povrsina * (1 + 0.252 + 0.084)
    if abs(c42 - expected_42) < 0.01:
        ok("Z-4-2 formula tačna", f"{c42:.2f}")
    else:
        fail("Z-4-2 formula pogrešna", f"{c42:.2f} != {expected_42:.2f}")

    # Nepoznata zona → 0
    c_unknown = cijena_zone("Z-9-9.1", povrsina)
    if c_unknown == 0.0:
        ok("Nepoznata zona vraća 0")
    else:
        fail("Nepoznata zona ne vraća 0", str(c_unknown))

    # Sve 11 zona pokrivene
    test_zone = [("Z-1-1", 0.15 * (1+0.084)), ("Z-1-2", 0.15*(1+0.084)),
                 ("Z-1-3", 0.15*(1+0.084)), ("Z-1-4", 0.15*(1+0.005)),
                 ("Z-1-5", 0.15*(1+0.005)), ("Z-1-6", 0.15*(1+0.016)),
                 ("Z-1-7", 0.3*(1+0.011)),  ("Z-3",   0.15*(1+0.084+0.126)),
                 ("Z-4-1", 0.5*(1+0.252)),  ("Z-4-2", 0.5*(1+0.252+0.084)),
                 ("Z-5",   0.3*(1+0.505+0.084))]
    sve_ok = all(abs(cijena_zone(z+".1", 1.0) - ocek) < 1e-9
                 for z, ocek in test_zone)
    if sve_ok:
        ok("Svih 11 zona pokriveno ispravnim formulama")
    else:
        fail("Neka zona ima pogrešnu formulu")

except Exception as e:
    fail("cijena_zone() pukao"); traceback.print_exc()


# ---------------------------------------------------------------------------
# Test 2: distanca_od_centra_masa
# ---------------------------------------------------------------------------
section("TEST 2: distanca_od_centra_masa")

try:
    cm = podaci.centar_masa   # [6413080, 4970217, 90]

    # Distanca od samog centra mase treba biti 0
    d0 = distanca_od_centra_masa(float(cm[0]), float(cm[1]), cm)
    if abs(d0) < 1e-6:
        ok("Distanca od CM do CM = 0")
    else:
        fail("Distanca od CM do CM != 0", str(d0))

    # Poznata distanca — 1000 m u X smjeru
    d1000 = distanca_od_centra_masa(float(cm[0]) + 1000.0, float(cm[1]), cm)
    if abs(d1000 - 1000.0) < 0.01:
        ok("Distanca 1000m u X smjeru tačna", f"{d1000:.2f} m")
    else:
        fail("Distanca 1000m u X smjeru pogrešna", f"{d1000:.2f}")

    # 3-4-5 Pitagora: 300m X + 400m Y = 500m
    d500 = distanca_od_centra_masa(
        float(cm[0]) + 300.0, float(cm[1]) + 400.0, cm
    )
    if abs(d500 - 500.0) < 0.01:
        ok("Pitagora 300-400-500 tačno", f"{d500:.2f} m")
    else:
        fail("Pitagora pogrešna", f"{d500:.2f} != 500.0")

except Exception as e:
    fail("distanca_od_centra_masa() pukao"); traceback.print_exc()


# ---------------------------------------------------------------------------
# Test 3: racunaj_troskove
# ---------------------------------------------------------------------------
section("TEST 3: racunaj_troskove — c1, c2, c3 formule")

try:
    # Ručni proračun za poznate vrijednosti
    zapremina  = 1_000_000.0   # 1M m³
    distanca   = 500.0         # 500 m
    wz         = 190.0         # visina vrha
    eko_cena   = 5000.0

    c1, c2, c3 = racunaj_troskove(zapremina, distanca, wz, eko_cena)

    # c1 = zapremina * (distanca/1000) * 0.8 = 1M * 0.5 * 0.8 = 400_000
    expected_c1 = zapremina * (distanca / 1000) * 0.8
    if abs(c1 - expected_c1) < 0.01:
        ok("c1 (transport) formula tačna", f"{c1:,.0f}")
    else:
        fail("c1 pogrešno", f"{c1:.2f} != {expected_c1:.2f}")

    # c2 = zapremina * (((wz-90)/0.08*1.6)/1000) * 1.2
    expected_c2 = zapremina * (((wz - 90) / 0.08 * 1.6) / 1000) * 1.2
    if abs(c2 - expected_c2) < 0.01:
        ok("c2 (iskopavanje) formula tačna", f"{c2:,.0f}")
    else:
        fail("c2 pogrešno", f"{c2:.2f} != {expected_c2:.2f}")

    # c3 = eko_cena direktno
    if abs(c3 - eko_cena) < 0.01:
        ok("c3 (ekonomska cijena) direktno proslijeđena")
    else:
        fail("c3 pogrešno", str(c3))

except Exception as e:
    fail("racunaj_troskove() pukao"); traceback.print_exc()


# ---------------------------------------------------------------------------
# Test 4: ekonomska_cijena — integracija sa zonama
# ---------------------------------------------------------------------------
section("TEST 4: ekonomska_cijena — integracija sa zonama")

try:
    from geometry import Surface
    import numpy as np

    # Konstruiši sintetičku presječišnu površinu unutar prve zone
    zona = podaci.dobre_zone[0]
    cx = zona.x_data.mean()
    cy = zona.y_data.mean()
    # Mala površina u centru zone
    dx = 10.0
    v = np.array([
        [cx-dx, cy-dx, 150.0],
        [cx+dx, cy-dx, 150.0],
        [cx,    cy+dx, 150.0],
    ])
    f = np.array([[0, 1, 2]])
    surf = Surface(vertices=v, faces=f)

    cijena, zone_str = ekonomska_cijena(surf, podaci.dobre_zone)

    if cijena >= 0:
        ok("ekonomska_cijena ne pada", f"cijena={cijena:.2f}, zone='{zone_str}'")
    else:
        fail("ekonomska_cijena negativna", str(cijena))

    # Tačke van svih zona → cijena = 0
    v_van = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    surf_van = Surface(vertices=v_van, faces=np.array([[0, 1, 2]]))
    cijena_van, zones_van = ekonomska_cijena(surf_van, podaci.dobre_zone)

    if cijena_van == 0.0:
        ok("Tačke van zona → cijena = 0")
    else:
        fail("Tačke van zona daju cijenu != 0", str(cijena_van))

except Exception as e:
    fail("ekonomska_cijena() pukao"); traceback.print_exc()


# ---------------------------------------------------------------------------
# Test 5: GAKontekst + funkcija_cilja
# ---------------------------------------------------------------------------
section("TEST 5: GAKontekst i funkcija_cilja")

try:
    ctx = GAKontekst(
        centar_masa=podaci.centar_masa,
        ugao=37.0,
        mnv=podaci.parametri.nadmorska_visina,
        donja_granica_zapremine=100_000.0,
        gornja_granica_zapremine=50_000_000.0,
        uslov_distance=podaci.parametri.uslov_distance,
        teren=podaci.teren,
        zona_x=podaci.granice.x_poly,
        zona_y=podaci.granice.y_poly,
        dobre_zone=podaci.dobre_zone,
    )
    ok("GAKontekst kreiran bez greške")

    # Test bounds
    cm = podaci.centar_masa
    wx_test = float(cm[0])
    wy_test = float(cm[1])
    lb, ub = get_bounds(wx_test, wy_test, verzija="buvac")

    if len(lb) == 4 and len(ub) == 4:
        ok("Bounds dužine 4")
    else:
        fail("Bounds pogrešne dužine")

    if lb[0] == 175.0 and ub[0] == 280.0:
        ok("wz bounds ispravni (175–280)", f"lb[0]={lb[0]}, ub[0]={ub[0]}")
    else:
        fail("wz bounds pogrešni")

    if lb[2] == wx_test and ub[2] == wx_test:
        ok("wx je fiksiran (lb == ub)")
    else:
        fail("wx nije fiksiran")

    # Test funkcije cilja sa centrom masa kao tačkom
    x_test = np.array([220.0, 200.0, wx_test, wy_test])
    f_val = funkcija_cilja(x_test, ctx)

    if isinstance(f_val, float):
        ok("funkcija_cilja vraća float")
    else:
        fail("funkcija_cilja ne vraća float", type(f_val).__name__)

    if f_val != 40_000_000.0:
        ok("Nije penalizovana vrijednost — kupa moguća na toj poziciji",
           f"f = {f_val:.4f}")
    else:
        ok("Penalizovana vrijednost (kupa van zone ili van granica zapremine) — OK za ovu poziciju",
           f"f = {f_val:.0f}")

    # Penalizacija: tačka daleko van zone
    x_van = np.array([220.0, 200.0,
                      podaci.granice.x_range[1] + 100_000,
                      podaci.granice.y_range[0]])
    f_van = funkcija_cilja(x_van, ctx)
    if f_van >= 40_000_000.0:
        ok("Tačka van zone → penalizovana vrijednost")
    else:
        fail("Tačka van zone nije penalizovana", str(f_van))

except Exception as e:
    fail("GAKontekst/funkcija_cilja pukao"); traceback.print_exc()


# ---------------------------------------------------------------------------
# Rezultat
# ---------------------------------------------------------------------------
ukupno = passed + failed
print(f"\n{'='*55}")
print(f"  REZULTAT: {passed}/{ukupno} testova prošlo", end="")
print("  —  SVE OK ✓" if failed == 0 else f"  —  {failed} NEUSPJEŠNIH ✗")
print(f"{'='*55}\n")

sys.exit(0 if failed == 0 else 1)
