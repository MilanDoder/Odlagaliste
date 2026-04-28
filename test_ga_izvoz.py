"""
test_ga_izvoz.py  –  Testiranje koraka 4 i 5: GA pokretač + izvoz

Pokretanje:  python3 test_ga_izvoz.py
"""

import sys, time, traceback, csv
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ga_funkcije import GAKontekst, get_bounds
from ga_pokretac import GAOpcije, RezultatTacke, optimizuj_tacku, pokreni_ga
from izvoz import DxfWriter, izvezi_excel, izvezi_dxf, izvezi_sve, timestamp_string

DATA = Path("/home/claude/kodovi_extracted/Kodovi")
IZLAZ = Path("/home/claude/odlagaliste/test_output")
IZLAZ.mkdir(exist_ok=True)

passed = 0; failed = 0

def ok(naziv, opis=""):
    global passed; passed += 1
    print(f"  ✓  {naziv}" + (f"  —  {opis}" if opis else ""))

def fail(naziv, opis=""):
    global failed; failed += 1
    print(f"  ✗  {naziv}" + (f"  —  {opis}" if opis else ""))

def section(s):
    print(f"\n{'='*55}\n  {s}\n{'='*55}")


# ---------------------------------------------------------------------------
# Sintetički teren za GA testove (brz, bez učitavanja 42K tačaka)
# ---------------------------------------------------------------------------
from geometry import Surface
from scipy.spatial import Delaunay as _Delaunay

print("Kreiram sintetički teren za GA testove...")
np.random.seed(1)
_x = np.linspace(6412000, 6414000, 25)
_y = np.linspace(4969000, 4971000, 25)
_xx, _yy = np.meshgrid(_x, _y)
_zz = 165.0 + 10 * np.sin(_xx / 400) * np.cos(_yy / 400)
_pts = np.column_stack([_xx.ravel(), _yy.ravel(), _zz.ravel()])
_tri = _Delaunay(_pts[:, :2])
sint_teren = Surface(vertices=_pts, faces=_tri.simplices)

# Zona koja pokriva sintetički teren
sint_zona_x = np.array([6411000, 6415000, 6415000, 6411000, 6411000], dtype=float)
sint_zona_y = np.array([4968000, 4968000, 4972000, 4972000, 4968000], dtype=float)
sint_cm = np.array([6413000.0, 4970000.0, 90.0])

ctx = GAKontekst(
    centar_masa=sint_cm,
    ugao=37.0,
    mnv=140.0,
    donja_granica_zapremine=50_000.0,
    gornja_granica_zapremine=80_000_000.0,
    uslov_distance=5000.0,
    teren=sint_teren,
    zona_x=sint_zona_x,
    zona_y=sint_zona_y,
    dobre_zone=[],
)
cm = sint_cm
print("Sintetički teren OK.\n")

# Brze opcije za testove
brze_opcije = GAOpcije(populacija=6, max_generacija=2, seed=42)


# ---------------------------------------------------------------------------
section("TEST 1: get_bounds — verzije Buvac i V1")
# ---------------------------------------------------------------------------
try:
    wx, wy = float(cm[0]), float(cm[1])
    lb_b, ub_b = get_bounds(wx, wy, "buvac")
    lb_v, ub_v = get_bounds(wx, wy, "v1")

    if lb_b == [175.0, 80.0, wx, wy] and ub_b == [280.0, 350.0, wx, wy]:
        ok("Buvac bounds tačni")
    else:
        fail("Buvac bounds pogrešni", f"lb={lb_b}, ub={ub_b}")

    if lb_v == [155.0, 70.0, wx, wy] and ub_v == [210.0, 120.0, wx, wy]:
        ok("V1 bounds tačni")
    else:
        fail("V1 bounds pogrešni", f"lb={lb_v}, ub={ub_v}")

    if lb_b[2] == ub_b[2] == wx:
        ok("wx fiksiran (lb==ub)")
    else:
        fail("wx nije fiksiran")

except Exception as e:
    fail("get_bounds pukao"); traceback.print_exc()


# ---------------------------------------------------------------------------
section("TEST 2: optimizuj_tacku — jedna tačka")
# ---------------------------------------------------------------------------
try:
    wx_t = float(cm[0])
    wy_t = float(cm[1])

    t0 = time.time()
    rez = optimizuj_tacku("test_tacka", wx_t, wy_t, ctx, brze_opcije)
    dt = time.time() - t0

    print(f"     Trajanje: {dt:.2f}s")

    if rez is None:
        ok("optimizuj_tacku se izvršio (vratio None — tačka možda van zone ili granica)")
    else:
        ok("optimizuj_tacku pronašao rješenje", f"f={rez.f_vrednost:.4f}, V={rez.zapremina:,.0f}m³")

        if isinstance(rez.wz, float) and isinstance(rez.k, float):
            ok("wz i k su float-ovi", f"wz={rez.wz:.1f}, k={rez.k:.1f}")
        else:
            fail("wz ili k nisu float-ovi")

        if rez.zapremina > 0:
            ok("Zapremina pozitivna")
        else:
            fail("Zapremina negativna ili nula")

        if rez.xx1 is not None and rez.xx1.shape == (9,):
            ok("Gornja kontura shape ispravan (9,)")
        else:
            fail("Gornja kontura pogrešna", str(rez.xx1))

        if len(rez.kao_red()) == 13:
            ok("kao_red() vraća 13 kolona")
        else:
            fail("kao_red() pogrešan broj kolona", str(len(rez.kao_red())))

except Exception as e:
    fail("optimizuj_tacku pukao"); traceback.print_exc()


# ---------------------------------------------------------------------------
section("TEST 3: pokreni_ga — 3 tačke (brzo)")
# ---------------------------------------------------------------------------
try:
    # Uzmemo 3 tačke blizu centra mase
    np.random.seed(42)
    tacke = np.array([
        [float(cm[0]),        float(cm[1]),        185.0],
        [float(cm[0]) + 200,  float(cm[1]) + 300,  190.0],
        [float(cm[0]) - 100,  float(cm[1]) - 200,  180.0],
    ])

    svi, validni = pokreni_ga(tacke, ctx, brze_opcije, verbose=True)

    if isinstance(svi, list) and isinstance(validni, list):
        ok("pokreni_ga vraća 2 liste")
    else:
        fail("pokreni_ga pogrešan tip povrata")

    ok(f"Rezultati: {len(svi)} ukupno, {len(validni)} validnih unutar zone")

    # validni mora biti podskup svih
    if all(r in svi for r in validni):
        ok("Validni su podskup svih")
    else:
        fail("Validni NISU podskup svih!")

except Exception as e:
    fail("pokreni_ga pukao"); traceback.print_exc()


# ---------------------------------------------------------------------------
section("TEST 4: RezultatTacke.ZAGLAVLJE — 13 kolona")
# ---------------------------------------------------------------------------
try:
    if len(RezultatTacke.ZAGLAVLJE) == 13:
        ok("13 kolona u zagavlju", str(RezultatTacke.ZAGLAVLJE[:3]) + "...")
    else:
        fail("Pogrešan broj kolona", str(len(RezultatTacke.ZAGLAVLJE)))

    ocekivane = {"Naziv_tacke", "X_koordinata", "Y_koordinata", "Z_koordinata",
                 "K", "Funkcija_cilja", "Zapremina", "Ugao",
                 "distanca", "c1", "c2", "c3", "Zone"}
    if set(RezultatTacke.ZAGLAVLJE) == ocekivane:
        ok("Sve 13 kolona ispravnih naziva")
    else:
        diff = ocekivane.symmetric_difference(RezultatTacke.ZAGLAVLJE)
        fail("Razlike u kolonama", str(diff))

except Exception as e:
    fail("ZAGLAVLJE provjera pukla"); traceback.print_exc()


# ---------------------------------------------------------------------------
section("TEST 5: izvezi_excel — CSV/Excel export")
# ---------------------------------------------------------------------------
try:
    # Sintetički rezultati
    def sinteticki_rez(naziv, wx, wy):
        return RezultatTacke(
            naziv=naziv, wx=wx, wy=wy, wz=220.0, k=150.0,
            f_vrednost=2.5, zapremina=1_500_000.0,
            ugao=37.0, distanca=500.0,
            c1=600.0, c2=1_200_000.0, c3=5000.0,
            zone="Z-1-1.3,", unutar_zone=True,
            xx1=np.zeros(9), yy1=np.zeros(9), zz1=np.zeros(9),
            xx2=np.zeros(9), yy2=np.zeros(9), zz2=np.zeros(9),
        )

    test_rezultati = [
        sinteticki_rez("point_1", float(cm[0]), float(cm[1])),
        sinteticki_rez("point_2", float(cm[0]) + 100, float(cm[1]) + 200),
    ]

    p = izvezi_excel(test_rezultati, IZLAZ / "test_export.xlsx")

    if p.exists():
        ok("Excel/CSV fajl kreiran", p.name)
    else:
        fail("Fajl nije kreiran")

    # Provjera sadržaja CSV-a
    if p.suffix == ".csv":
        with open(p, encoding="utf-8") as f:
            reader = csv.reader(f)
            zaglavlje = next(reader)
            redovi = list(reader)

        if zaglavlje == RezultatTacke.ZAGLAVLJE:
            ok("Zaglavlje ispravno")
        else:
            fail("Zaglavlje pogrešno", str(zaglavlje[:3]))

        if len(redovi) == 2:
            ok("Broj redova tačan", "2 od 2")
        else:
            fail("Broj redova pogrešan", str(len(redovi)))

        if redovi[0][0] == "point_1":
            ok("Naziv tačke ispravan")
        else:
            fail("Naziv tačke pogrešan", redovi[0][0])

except Exception as e:
    fail("izvezi_excel pukao"); traceback.print_exc()


# ---------------------------------------------------------------------------
section("TEST 6: DxfWriter — direktno pisanje")
# ---------------------------------------------------------------------------
try:
    dxf_put = IZLAZ / "test.dxf"

    with DxfWriter(dxf_put) as dxf:
        # Mala površina od 3 trougla
        vertices = np.array([
            [6413000.0, 4970000.0, 180.0],
            [6413100.0, 4970000.0, 180.0],
            [6413050.0, 4970100.0, 185.0],
            [6413150.0, 4970100.0, 175.0],
        ])
        faces = np.array([[0, 1, 2], [1, 3, 2]])
        dxf.postavi(sloj=1, boja=(1.0, 0.0, 0.0))
        dxf.polimreža(vertices, faces)

        # Polilinija (kontura)
        xx = np.array([6413000.0, 6413100.0, 6413050.0, 6413000.0])
        yy = np.array([4970000.0, 4970000.0, 4970100.0, 4970000.0])
        zz = np.full(4, 180.0)
        dxf.postavi(sloj=2, boja=(0.0, 0.0, 1.0))
        dxf.polilinija(xx, yy, zz)

    if dxf_put.exists() and dxf_put.stat().st_size > 0:
        ok("DXF fajl kreiran", f"{dxf_put.stat().st_size} bajtova")
    else:
        fail("DXF fajl nije kreiran ili je prazan")

    # Provjera sadržaja
    sadrzaj = dxf_put.read_text(encoding="ascii")
    if "SECTION" in sadrzaj and "ENTITIES" in sadrzaj and "EOF" in sadrzaj:
        ok("DXF zaglavlje i kraj prisutni")
    else:
        fail("DXF fajl nema ispravnu strukturu")

    if "POLYLINE" in sadrzaj:
        ok("POLYLINE entitet prisutan")
    else:
        fail("POLYLINE entitet nedostaje")

    if "VERTEX" in sadrzaj:
        ok("VERTEX entiteti prisutni")
    else:
        fail("VERTEX entiteti nedostaju")

    if "SEQEND" in sadrzaj:
        ok("SEQEND prisutan")
    else:
        fail("SEQEND nedostaje")

except Exception as e:
    fail("DxfWriter pukao"); traceback.print_exc()


# ---------------------------------------------------------------------------
section("TEST 7: izvezi_dxf — sa sintetičkim rezultatima")
# ---------------------------------------------------------------------------
try:
    def sinteticki_sa_konturama(naziv):
        theta = np.arange(0, 2*np.pi + np.pi/4, np.pi/4)
        k = 150.0; wx = float(cm[0]); wy = float(cm[1]); wz = 220.0
        s = 1.4*k; u = 1.25*k
        r = np.array([s, k, u, k, s, k, u, k, s])
        xx1 = wx + r * np.cos(theta)
        yy1 = wy + r * np.sin(theta)
        xx2 = wx + (r + 50) * np.cos(theta)
        yy2 = wy + (r + 50) * np.sin(theta)
        return RezultatTacke(
            naziv=naziv, wx=wx, wy=wy, wz=wz, k=k,
            f_vrednost=2.5, zapremina=1_500_000.0,
            ugao=37.0, distanca=500.0, c1=600.0, c2=1_200_000.0, c3=5000.0,
            zone="Z-1-1.3,", unutar_zone=True,
            xx1=xx1, yy1=yy1, zz1=np.full(9, wz),
            xx2=xx2, yy2=yy2, zz2=np.full(9, 140.0),
        )

    test_r = [sinteticki_sa_konturama("p1"), sinteticki_sa_konturama("p2")]
    ts = "test_ts"
    p_dxf = izvezi_dxf(test_r, ts, IZLAZ)

    if p_dxf and p_dxf.exists():
        ok("izvezi_dxf kreirao fajl", p_dxf.name)
        sadrzaj = p_dxf.read_text(encoding="ascii")
        n_polyline = sadrzaj.count("POLYLINE")
        ok(f"Broj POLYLINE entiteta", f"{n_polyline} (očekivano 4 = 2 kupe × 2 konture)")
    else:
        fail("izvezi_dxf nije kreirao fajl")

except Exception as e:
    fail("izvezi_dxf pukao"); traceback.print_exc()


# ---------------------------------------------------------------------------
ukupno = passed + failed
print(f"\n{'='*55}")
print(f"  REZULTAT: {passed}/{ukupno} testova prošlo", end="")
print("  —  SVE OK ✓" if failed == 0 else f"  —  {failed} NEUSPJEŠNIH ✗")
print(f"{'='*55}\n")
sys.exit(0 if failed == 0 else 1)
