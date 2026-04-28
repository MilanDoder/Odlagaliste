"""
ga_funkcije.py  –  Korak 3b: funkcije genetskog algoritma

Zamjenjuje MATLAB funkcije:
  funkcijaCiljaGenetskiAlgoritam.m       → funkcija_cilja()
  funkacijaOgranicenjaGenetskiAlgoritam.m → funkcija_ogranicenja()

GA varijable (4 varijable, identično MATLAB kodu):
  x[0] = wz   — visina vrha kupe (npr. 175–280 m)
  x[1] = k    — širina kupe (npr. 80–350 m)
  x[2] = wx   — X koordinata (fiksna — centar tačke iz Monte Carlo)
  x[3] = wy   — Y koordinata (fiksna — centar tačke iz Monte Carlo)

Bounds iz IzvrsniKodBuvac.m:
  Donja granica: [175, 80,  pointX, pointY]
  Gornja granica: [280, 350, pointX, pointY]

Napomena: wx i wy su u MATLAB kodu bili fiksirani (lb==ub za te dvije var).
Ovdje to modeliramo kao pojedinačan par koordinata koji GA optimizuje
samo wz i k.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from geometry import zapremina_kupe, unutar_interesne_zone, Surface
from ekonomija import distanca_od_centra_masa, ekonomska_cijena, racunaj_troskove


# ---------------------------------------------------------------------------
# Kontekst GA  —  sve što funkcije trebaju, bez globalnih varijabli
# ---------------------------------------------------------------------------

@dataclass
class GAKontekst:
    """Zamjenjuje sve globalne varijable koje MATLAB GA funkcije čitaju.

    U MATLAB kodu getCentarMasa(), getUgao(), getDonjaGranicaZapremineOdlagalista()
    itd. su čitali globalne varijable. Ovdje se sve prosleđuje eksplicitno.
    """
    centar_masa: np.ndarray        # [X, Y, Z]
    ugao: float                    # ugao kosine (stepeni)
    mnv: float                     # nadmorska visina baze (DodatniParametri.nadmorska_visina)
    donja_granica_zapremine: float # min zapremina odlagališta (m³)
    gornja_granica_zapremine: float # max zapremina odlagališta (m³)
    uslov_distance: float          # max dozvoljena distanca od centra masa (m)
    teren: "Surface"               # triangulisana površina terena
    zona_x: np.ndarray             # poligon interesne zone X
    zona_y: np.ndarray             # poligon interesne zone Y
    dobre_zone: list               # lista EkonomskaZona


# ---------------------------------------------------------------------------
# Bounds definicija  (direktno iz MATLAB koda)
# ---------------------------------------------------------------------------

def get_bounds(
    wx: float, wy: float,
    verzija: str = "buvac",
) -> tuple[list[float], list[float]]:
    """Vraća donje i gornje granice za GA varijable.

    MATLAB ekvivalent (IzvrsniKodBuvac.m):
        lb = [175, 80,  pointX, pointY]
        ub = [280, 350, pointX, pointY]

    MATLAB ekvivalent (IzvrsniKod.m — starija verzija):
        lb = [155, 70,  pointX, pointY]
        ub = [210, 120, pointX, pointY]

    Args:
        wx, wy:   koordinate tačke (fiksirani bounds za var 2 i 3)
        verzija:  "buvac" (IzvrsniKodBuvac) ili "v1" (IzvrsniKod)

    Returns:
        (lb, ub) — liste dužine 4
    """
    if verzija == "buvac":
        lb = [175.0, 80.0,  wx, wy]
        ub = [280.0, 350.0, wx, wy]
    else:  # v1
        lb = [155.0, 70.0,  wx, wy]
        ub = [210.0, 120.0, wx, wy]
    return lb, ub


# ---------------------------------------------------------------------------
# Funkcija cilja  (zamjenjuje funkcijaCiljaGenetskiAlgoritam.m)
# ---------------------------------------------------------------------------

def funkcija_cilja(x: np.ndarray, ctx: GAKontekst) -> float:
    """Funkcija cilja genetskog algoritma — minimizuje trošak/zapreminu.

    MATLAB ekvivalent: funkcijaCiljaGenetskiAlgoritam(x)

    GA varijable (x — array dužine 4):
        x[0] = wz   — visina vrha kupe
        x[1] = k    — širina kupe
        x[2] = wx   — X koordinata (fiksna)
        x[3] = wy   — Y koordinata (fiksna)

    Funkcija cilja (iz MATLAB koda):
        distanca = pdist([centarMasa(1:2); [px, py]], 'euclidean')
        c1 = (distanca/1000) * 0.8 * zapremina
        c2 = zapremina * (((wz - 90) / 0.08 * 1.6) / 1000) * 1.2
        f  = (c1 + c2 + ekonomska_cijena) / zapremina

    Napomena: GA minimizuje f, pa manji f = bolja lokacija.
    Penalizacija: ako zapremina nije u granicama → vraća veliki broj.

    Args:
        x:   array [wz, k, wx, wy]
        ctx: GAKontekst sa svim parametrima

    Returns:
        Vrijednost funkcije cilja (skalarna)
    """
    wz = float(x[0])
    k  = float(x[1])
    wx = float(x[2])
    wy = float(x[3])

    # Ekonomska cijena se računa unutar zapremina_kupe
    def eko_fn(surf: Surface) -> tuple[float, str]:
        return ekonomska_cijena(surf, ctx.dobre_zone)

    rez = zapremina_kupe(
        wx=wx, wy=wy, wz=wz,
        ugao=ctx.ugao, k=k,
        mnv=ctx.mnv,
        teren=ctx.teren,
        zona_x=ctx.zona_x,
        zona_y=ctx.zona_y,
        ekonomska_fn=eko_fn,
    )

    zapremina = rez.zapremina

    # Penalizacija ako zapremina nije izračunata
    if zapremina >= 40_000_000:
        return 40_000_000.0

    # Penalizacija ako je van granica zapremine
    if (zapremina < ctx.donja_granica_zapremine or
            zapremina > ctx.gornja_granica_zapremine):
        return 40_000_000.0

    distanca = distanca_od_centra_masa(wx, wy, ctx.centar_masa)

    c1 = (distanca / 1000) * 0.8 * zapremina
    c2 = zapremina * (((wz - 90) / 0.08 * 1.6) / 1000) * 1.2
    eko = rez.ekonomska_cena

    if zapremina < 1e-6:
        return 40_000_000.0

    return (c1 + c2 + eko) / zapremina


# ---------------------------------------------------------------------------
# Funkcija ograničenja  (zamjenjuje funkacijaOgranicenjaGenetskiAlgoritam.m)
# ---------------------------------------------------------------------------

def funkcija_ogranicenja(x: np.ndarray, ctx: GAKontekst) -> np.ndarray:
    """Nelinearna ograničenja za GA — vraća c gdje je c <= 0 dopustivo.

    MATLAB ekvivalent: funkacijaOgranicenjaGenetskiAlgoritam(x)

    MATLAB c vektor:
        c(1) = donjaGranZapremina - V      <= 0  → zapremina >= donja granica
        c(2) = V - gornjaGranZapremina     <= 0  → zapremina <= gornja granica
        c(3) = distanca - distancaTransporta <= 0  → distanca <= max dozvoljena

    Args:
        x:   array [wz, k, wx, wy]
        ctx: GAKontekst

    Returns:
        np.ndarray dužine 3 — vrijednosti ograničenja (c <= 0 = dopustivo)
    """
    wz = float(x[0])
    k  = float(x[1])
    wx = float(x[2])
    wy = float(x[3])

    rez = zapremina_kupe(
        wx=wx, wy=wy, wz=wz,
        ugao=ctx.ugao, k=k,
        mnv=ctx.mnv,
        teren=ctx.teren,
        zona_x=ctx.zona_x,
        zona_y=ctx.zona_y,
    )

    V = rez.zapremina
    distanca = distanca_od_centra_mas(wx, wy, ctx.centar_masa)

    c = np.array([
        ctx.donja_granica_zapremine - V,        # c[0] <= 0
        V - ctx.gornja_granica_zapremine,        # c[1] <= 0
        distanca - ctx.uslov_distance,           # c[2] <= 0
    ])
    return c


# ---------------------------------------------------------------------------
# Priprema wrapper-a za scipy.optimize differential_evolution / DEAP
# ---------------------------------------------------------------------------

def napravi_fitness_fn(ctx: GAKontekst) -> Callable[[np.ndarray], float]:
    """Vraća fitness funkciju zatvorenu nad kontekstom.

    Korisno za scipy.optimize.differential_evolution koji prima
    fn(x) → float umjesto fn(x, ctx) → float.
    """
    def fitness(x: np.ndarray) -> float:
        return funkcija_cilja(x, ctx)
    return fitness


def napravi_ogranicenja_fn(ctx: GAKontekst):
    """Vraća listu ograničenja u scipy formatu.

    scipy.optimize.differential_evolution prima constraints kao:
        [{'type': 'ineq', 'fun': fn}]  gdje fn(x) >= 0 znači dopustivo.

    MATLAB c <= 0 = dopustivo → scipy fun >= 0 = dopustivo → negiramo.
    """
    def c1(x): return -(ctx.donja_granica_zapremine - _V(x, ctx))  # V >= donja
    def c2(x): return -(  _V(x, ctx) - ctx.gornja_granica_zapremine)  # V <= gornja
    def c3(x): return -(distanca_od_centra_masa(x[2], x[3], ctx.centar_masa) - ctx.uslov_distance)

    return [
        {"type": "ineq", "fun": c1},
        {"type": "ineq", "fun": c2},
        {"type": "ineq", "fun": c3},
    ]


def _V(x: np.ndarray, ctx: GAKontekst) -> float:
    """Pomoćna funkcija — vraća zapreminu za dati x vektor."""
    rez = zapremina_kupe(
        wx=float(x[2]), wy=float(x[3]), wz=float(x[0]),
        ugao=ctx.ugao, k=float(x[1]),
        mnv=ctx.mnv,
        teren=ctx.teren,
        zona_x=ctx.zona_x,
        zona_y=ctx.zona_y,
    )
    return rez.zapremina


# ---------------------------------------------------------------------------
# Ispravka typa — MATLAB kod ima distancaOdCentraMasa s 4 arg, naš ima 3
# ---------------------------------------------------------------------------
def distanca_od_centra_mas(x, y, centar_masa):
    """Alias koji prihvata (x, y, cm) — identičan interfejs kao MATLAB."""
    return distanca_od_centra_masa(x, y, centar_masa)
