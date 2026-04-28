"""
main.py  –  Glavni pokretač optimizacije odlagališta

Potpuna zamjena za MATLAB IzvrsniKodBuvac.m

Tok izvršavanja (identičan MATLAB-u):
  1.  Učitaj sve ulazne podatke (bez uigetfile dijaloga)
  2.  Vizualizacija terena (opcionalno)
  3.  Monte Carlo: generiši nasumične kandidat-tačke, filtriraj po zonama
  4.  Za svaku dobru tačku: provjeri geometriju kupe (presječište s terenom)
  5.  Pokreni GA za svaku preživjelu tačku
  6.  Post-procesiranje: ekonomski proračun, validacija unutar zone
  7.  Izvoz: Excel (svi + validni) + DXF

Pokretanje:
  python3 main.py --config config.json
  python3 main.py --teren teren.txt --zone zone.txt --cm cm.txt \
                  --granice granica.txt --params parametri.txt \
                  --tacke 200 --ponavljanja 3 --ugao 37 \
                  --min-v 100000 --max-v 50000000 --verzija buvac

Ili interaktivno (pita za parametre kao MATLAB):
  python3 main.py --interaktivno
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Moduli projekta
# ---------------------------------------------------------------------------
from loaders import ucitaj_sve, UlazniPodaci
from geometry import generiši_tačke, zapremina_kupe
from ekonomija import ekonomska_cijena, distanca_od_centra_masa, racunaj_troskove
from ga_funkcije import GAKontekst, get_bounds
from ga_pokretac import GAOpcije, RezultatTacke, optimizuj_tacku, pokreni_ga
from izvoz import izvezi_sve, timestamp_string


# ---------------------------------------------------------------------------
# Argumenti komandne linije
# ---------------------------------------------------------------------------

def parsiraj_argumente() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optimizacija lokacije odlagališta — zamjena za IzvrsniKodBuvac.m"
    )
    p.add_argument("--config",      type=Path, help="JSON config fajl (sve opcije)")
    p.add_argument("--teren",       type=Path, help="Putanja do fajla terena (XYZ)")
    p.add_argument("--zone",        type=Path, help="Putanja do fajla ekonomskih zona")
    p.add_argument("--cm",          type=Path, help="Putanja do fajla centra masa")
    p.add_argument("--granice",     type=Path, help="Putanja do fajla granica zone")
    p.add_argument("--params",      type=Path, help="Putanja do fajla dodatnih parametara")
    p.add_argument("--tacke",       type=int,   default=100,  help="Broj Monte Carlo tačaka")
    p.add_argument("--ponavljanja", type=int,   default=1,    help="Broj ponavljanja")
    p.add_argument("--ugao",        type=float, default=37.0, help="Ugao kosine (stepeni)")
    p.add_argument("--min-v",       type=float, default=100_000.0,   help="Min zapremina (m³)")
    p.add_argument("--max-v",       type=float, default=50_000_000.0, help="Max zapremina (m³)")
    p.add_argument("--verzija",     type=str,   default="buvac",
                   choices=["buvac", "v1"], help="Bounds verzija (buvac ili v1)")
    p.add_argument("--izlaz",       type=Path,  default=Path("rezultati"),
                   help="Izlazni direktorij")
    p.add_argument("--populacija",  type=int,   default=30,   help="GA populacija")
    p.add_argument("--interaktivno", action="store_true",
                   help="Unos parametara kroz konzolu (kao MATLAB)")
    p.add_argument("--bez-dxf",    action="store_true", help="Preskoči DXF izvoz")
    p.add_argument("--tiho",       action="store_true", help="Minimiziraj ispise")
    return p.parse_args()


def učitaj_config(putanja: Path) -> dict:
    """Učitava JSON config fajl."""
    return json.loads(putanja.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Interaktivni unos  (ekvivalent MATLAB input() poziva)
# ---------------------------------------------------------------------------

def interaktivni_unos() -> dict:
    """Pita korisnika za parametre kroz konzolu — identično MATLAB-u."""
    print("\n" + "="*55)
    print("  Optimizacija odlagališta — unos parametara")
    print("="*55 + "\n")

    def pitaj(poruka: str, tip=float, default=None):
        sufiks = f" [{default}]" if default is not None else ""
        while True:
            try:
                unos = input(f"{poruka}{sufiks}: ").strip()
                if not unos and default is not None:
                    return default
                return tip(unos)
            except ValueError:
                print(f"  Neispravan unos, pokušaj ponovo.")

    def pitaj_putanju(poruka: str) -> Path:
        while True:
            unos = input(f"{poruka}: ").strip()
            p = Path(unos)
            if p.exists():
                return p
            print(f"  Fajl nije pronađen: {p}")

    return {
        "teren":       str(pitaj_putanju("Putanja do fajla terena")),
        "zone":        str(pitaj_putanju("Putanja do fajla ekonomskih zona")),
        "cm":          str(pitaj_putanju("Putanja do fajla centra masa")),
        "granice":     str(pitaj_putanju("Putanja do fajla granica zone")),
        "params":      str(pitaj_putanju("Putanja do fajla dodatnih parametara")),
        "tacke":       pitaj("Broj nasumičnih tačaka", int, 100),
        "ugao":        pitaj("Ugao kosine odlagališta (stepeni)", float, 37.0),
        "min_v":       pitaj("Donja granica zapremine (m³)", float, 100_000.0),
        "max_v":       pitaj("Gornja granica zapremine (m³)", float, 50_000_000.0),
        "ponavljanja": pitaj("Broj ponavljanja Monte Carlo", int, 1),
        "populacija":  pitaj("GA populacija", int, 30),
    }


# ---------------------------------------------------------------------------
# Faza 3: Monte Carlo filtriranje + geometrijska provjera
# ---------------------------------------------------------------------------

def monte_carlo_faza(
    podaci: UlazniPodaci,
    ctx: GAKontekst,
    n_tacaka: int,
    n_ponavljanja: int,
    tiho: bool = False,
) -> list[tuple[str, float, float, float]]:
    """Generiše i filtrira kandidat-tačke.

    MATLAB ekvivalent: petlja numberRepeat + inpolygon + SurfaceIntersection + convhull blok

    Returns:
        lista (naziv, wx, wy, wz) dobrih tačaka
    """
    granice = podaci.granice
    dobre: list[tuple[str, float, float, float]] = []
    ukupno_gen = 0

    for rep in range(1, n_ponavljanja + 1):
        if not tiho:
            print(f"\n  Ponavljanje {rep}/{n_ponavljanja}: generišem {n_tacaka} tačaka...")

        tacke = generiši_tačke(
            n=n_tacaka,
            x_range=granice.x_range,
            y_range=granice.y_range,
            z_range=granice.z_range,
            zona_x=granice.x_poly,
            zona_y=granice.y_poly,
            lose_zone=podaci.lose_zone,
        )
        ukupno_gen += len(tacke)

        if not tiho:
            print(f"  Filtrirane tačke unutar zone: {len(tacke)}")
            print(f"  Provjera geometrije kupe za svaku tačku...")

        provjere_ok = 0
        for j, tacka in enumerate(tacke):
            wx, wy, wz = float(tacka[0]), float(tacka[1]), float(tacka[2])

            # Provjeri geometriju kupe (MATLAB: SurfaceIntersection + convhull blok)
            rez = zapremina_kupe(
                wx=wx, wy=wy, wz=wz,
                ugao=ctx.ugao, k=0.001,   # mali k za inicijalni test
                mnv=ctx.mnv,
                teren=podaci.teren,
                zona_x=granice.x_poly,
                zona_y=granice.y_poly,
            )

            if rez.zapremina < 40_000_000 and rez.intersect_surface is not None:
                if rez.intersect_surface.vertices.shape[0] > 0:
                    naziv = f"point_{j+1}_{rep}"
                    dobre.append((naziv, wx, wy, wz))
                    provjere_ok += 1

        if not tiho:
            print(f"  Tačaka sa presječištem terena: {provjere_ok}")

    if not tiho:
        print(f"\n  Ukupno generisano: {ukupno_gen}, dobrih za GA: {len(dobre)}")

    return dobre


# ---------------------------------------------------------------------------
# Glavni tok
# ---------------------------------------------------------------------------

def main():
    args = parsiraj_argumente()
    start_total = time.time()

    # --- Učitaj config ---
    cfg: dict = {}
    if args.config:
        cfg = učitaj_config(args.config)
    elif args.interaktivno:
        cfg = interaktivni_unos()

    def get(kljuc: str, attr: str, default):
        """Prioritet: config > arg > default"""
        if kljuc in cfg:
            return cfg[kljuc]
        val = getattr(args, attr, None)
        return val if val is not None else default

    # Putanje do fajlova
    p_teren   = Path(get("teren",   "teren",   None))
    p_zone    = Path(get("zone",    "zone",    None))
    p_cm      = Path(get("cm",      "cm",      None))
    p_granice = Path(get("granice", "granice", None))
    p_params  = Path(get("params",  "params",  None))

    for p, ime in [(p_teren, "teren"), (p_zone, "zone"),
                   (p_cm, "centar masa"), (p_granice, "granice"),
                   (p_params, "parametri")]:
        if p is None or not p.exists():
            print(f"GREŠKA: Fajl '{ime}' nije pronađen: {p}", file=sys.stderr)
            print("Pokrenite sa --interaktivno ili navedite putanje.", file=sys.stderr)
            sys.exit(1)

    n_tacaka     = int(get("tacke",       "tacke",       100))
    n_ponavljanja = int(get("ponavljanja", "ponavljanja", 1))
    ugao         = float(get("ugao",       "ugao",        37.0))
    min_v        = float(get("min_v",      "min_v",       100_000.0))
    max_v        = float(get("max_v",      "max_v",       50_000_000.0))
    verzija      = str(get("verzija",      "verzija",     "buvac"))
    izlaz_dir    = Path(get("izlaz",       "izlaz",       "rezultati"))
    populacija   = int(get("populacija",   "populacija",  30))
    tiho         = bool(get("tiho",        "tiho",        False))
    bez_dxf      = bool(get("bez_dxf",     "bez_dxf",     False))

    # -----------------------------------------------------------------------
    # Faza 1: Učitavanje podataka
    # -----------------------------------------------------------------------
    if not tiho:
        print("\n" + "="*55)
        print("  FAZA 1: Učitavanje podataka")
        print("="*55)

    podaci = ucitaj_sve(p_teren, p_zone, p_cm, p_granice, p_params)

    # Broj generacija GA — iz DodatniUlazniParametri ili default
    broj_gen = podaci.parametri.broj_generacija

    # -----------------------------------------------------------------------
    # Kontekst koji dijele sve funkcije
    # -----------------------------------------------------------------------
    ctx = GAKontekst(
        centar_masa=podaci.centar_masa,
        ugao=ugao,
        mnv=podaci.parametri.nadmorska_visina,
        donja_granica_zapremine=min_v,
        gornja_granica_zapremine=max_v,
        uslov_distance=podaci.parametri.uslov_distance,
        teren=podaci.teren,
        zona_x=podaci.granice.x_poly,
        zona_y=podaci.granice.y_poly,
        dobre_zone=podaci.dobre_zone,
    )

    opcije = GAOpcije(
        populacija=populacija,
        max_generacija=broj_gen,
    )

    # -----------------------------------------------------------------------
    # Faza 2: Monte Carlo + geometrijska provjera
    # -----------------------------------------------------------------------
    if not tiho:
        print("\n" + "="*55)
        print("  FAZA 2: Monte Carlo generisanje tačaka")
        print("="*55)

    t_mc = time.time()
    dobre_tacke = monte_carlo_faza(podaci, ctx, n_tacaka, n_ponavljanja, tiho)
    if not tiho:
        print(f"  Monte Carlo završen za {time.time()-t_mc:.1f}s")

    if not dobre_tacke:
        print("\nNema dobrih tačaka za GA. Povećaj broj tačaka ili provjeri podatke.")
        sys.exit(0)

    # -----------------------------------------------------------------------
    # Faza 3: Genetski algoritam
    # -----------------------------------------------------------------------
    if not tiho:
        print("\n" + "="*55)
        print("  FAZA 3: Genetski algoritam")
        print("="*55)

    # Pretvori dobre tačke u numpy array
    tacke_array = np.array([[wx, wy, wz] for _, wx, wy, wz in dobre_tacke])

    t_ga = time.time()
    svi_rezultati, validni_rezultati = pokreni_ga(
        tacke=tacke_array,
        ctx=ctx,
        opcije=opcije,
        verzija=verzija,
        verbose=not tiho,
    )
    if not tiho:
        print(f"  GA završen za {time.time()-t_ga:.1f}s")

    # -----------------------------------------------------------------------
    # Faza 4: Izvoz
    # -----------------------------------------------------------------------
    if not tiho:
        print("\n" + "="*55)
        print("  FAZA 4: Izvoz rezultata")
        print("="*55)

    izlaz_dir.mkdir(parents=True, exist_ok=True)

    if bez_dxf:
        from izvoz import izvezi_oba_excela, timestamp_string
        ts = timestamp_string()
        izvezi_oba_excela(svi_rezultati, validni_rezultati, izlaz_dir, ts)
        fajlovi = {"excel_svi": True, "excel_validni": True, "dxf": None}
    else:
        fajlovi = izvezi_sve(svi_rezultati, validni_rezultati, izlaz_dir)

    # -----------------------------------------------------------------------
    # Sažetak
    # -----------------------------------------------------------------------
    elapsed = time.time() - start_total
    print("\n" + "="*55)
    print("  ZAVRŠENO")
    print("="*55)
    print(f"  Ukupno trajanje:   {elapsed:.1f}s")
    print(f"  Monte Carlo tačke: {len(dobre_tacke)}")
    print(f"  GA rezultata:      {len(svi_rezultati)}")
    print(f"  Validnih (final):  {len(validni_rezultati)}")
    print(f"  Izlazni direktorij: {izlaz_dir.resolve()}")

    if validni_rezultati:
        # Ispiši top 3 po funkciji cilja
        sortirani = sorted(validni_rezultati, key=lambda r: r.f_vrednost)
        print(f"\n  Top {min(3, len(sortirani))} lokacija (najmanji f):")
        print(f"  {'Naziv':<18} {'X':>12} {'Y':>12} {'wz':>8} {'k':>8} {'f':>10} {'V (m³)':>14}")
        print(f"  {'-'*84}")
        for r in sortirani[:3]:
            print(f"  {r.naziv:<18} {r.wx:>12.0f} {r.wy:>12.0f} "
                  f"{r.wz:>8.1f} {r.k:>8.1f} {r.f_vrednost:>10.4f} {r.zapremina:>14,.0f}")
    print()


if __name__ == "__main__":
    main()
