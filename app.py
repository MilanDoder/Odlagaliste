"""
app.py  –  Streamlit web sučelje za optimizaciju odlagališta

Zamjenjuje:
  - MATLAB konzolni input()       → Streamlit forme i slideri
  - uigetfile() dijaloži          → file upload widgeti
  - trisurf() / MATLAB figure     → Plotly 3D interaktivni grafikoni
  - Progress nema u MATLAB        → st.progress() + st.status()

Pokretanje:
  streamlit run app.py
"""

from __future__ import annotations

import io
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

# Streamlit — ako nije instaliran, daje jasnu poruku
try:
    import streamlit as st
except ImportError:
    print("Instalacija: pip install streamlit plotly")
    sys.exit(1)

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

# Projektni moduli
sys.path.insert(0, str(Path(__file__).parent))
from loaders import (
    ucitaj_teren, ucitaj_ekonomske_zone, ucitaj_centar_masa,
    ucitaj_granice_zone, ucitaj_dodatne_parametre,
    UlazniPodaci, DodatniParametri,
)
from geometry import generiši_tačke, zapremina_kupe
from ekonomija import ekonomska_cijena
from ga_funkcije import GAKontekst
from ga_pokretac import GAOpcije, RezultatTacke, optimizuj_tacku
from izvoz import DxfWriter, izvezi_excel, timestamp_string


# ---------------------------------------------------------------------------
# Pomoćne funkcije
# ---------------------------------------------------------------------------

def spremi_upload(uploaded_file) -> Path:
    """Sprema Streamlit UploadedFile u privremeni fajl i vraća putanju."""
    suffix = Path(uploaded_file.name).suffix or ".txt"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return Path(tmp.name)


def plotly_teren_2d(teren_vertices: np.ndarray, naslov: str = "Teren") -> go.Figure:
    """2D scatter prikaz tačaka terena s bojom po visini."""
    fig = go.Figure(go.Scattergl(
        x=teren_vertices[:, 0],
        y=teren_vertices[:, 1],
        mode="markers",
        marker=dict(
            size=2,
            color=teren_vertices[:, 2],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Visina (m)"),
        ),
        hovertemplate="X: %{x:.0f}<br>Y: %{y:.0f}<br>Z: %{customdata:.1f}m",
        customdata=teren_vertices[:, 2],
        name="Teren",
    ))
    fig.update_layout(
        title=naslov,
        xaxis_title="X koordinata",
        yaxis_title="Y koordinata",
        height=450,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def plotly_rezultati(
    rezultati: list[RezultatTacke],
    teren_vertices: np.ndarray,
    zona_x: np.ndarray,
    zona_y: np.ndarray,
) -> go.Figure:
    """Interaktivna 2D mapa rezultata sa tačkama odlagališta."""
    # Teren u pozadini (decimirani)
    korak = max(1, len(teren_vertices) // 2000)
    t = teren_vertices[::korak]

    fig = go.Figure()

    # Teren
    fig.add_trace(go.Scattergl(
        x=t[:, 0], y=t[:, 1],
        mode="markers",
        marker=dict(size=2, color=t[:, 2], colorscale="Greys", opacity=0.4),
        name="Teren",
        showlegend=True,
    ))

    # Granica zone interesa
    fig.add_trace(go.Scatter(
        x=np.append(zona_x, zona_x[0]),
        y=np.append(zona_y, zona_y[0]),
        mode="lines",
        line=dict(color="#378ADD", width=2, dash="dash"),
        name="Granica zone",
    ))

    if rezultati:
        boje = ["#1D9E75" if r.unutar_zone else "#E24B4A" for r in rezultati]
        fig.add_trace(go.Scatter(
            x=[r.wx for r in rezultati],
            y=[r.wy for r in rezultati],
            mode="markers+text",
            marker=dict(size=12, color=boje, line=dict(width=1, color="white")),
            text=[f"{r.naziv}<br>f={r.f_vrednost:.3f}<br>V={r.zapremina/1e6:.2f}Mm³"
                  for r in rezultati],
            textposition="top center",
            name="Optimizovane tačke",
            hovertemplate=(
                "<b>%{text}</b><br>"
                "X: %{x:.0f}<br>Y: %{y:.0f}"
            ),
        ))

    fig.update_layout(
        title="Rezultati optimizacije",
        xaxis_title="X koordinata",
        yaxis_title="Y koordinata",
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def plotly_kupa_3d(rez: RezultatTacke) -> go.Figure:
    """3D prikaz konture jedne kupe."""
    fig = go.Figure()

    if rez.xx1 is not None:
        # Gornja kontura
        fig.add_trace(go.Scatter3d(
            x=np.append(rez.xx1, rez.xx1[0]),
            y=np.append(rez.yy1, rez.yy1[0]),
            z=np.append(rez.zz1, rez.zz1[0]),
            mode="lines",
            line=dict(color="#1D9E75", width=4),
            name="Gornja kontura",
        ))

    if rez.xx2 is not None:
        # Donja kontura
        fig.add_trace(go.Scatter3d(
            x=np.append(rez.xx2, rez.xx2[0]),
            y=np.append(rez.yy2, rez.yy2[0]),
            z=np.append(rez.zz2, rez.zz2[0]),
            mode="lines",
            line=dict(color="#378ADD", width=4),
            name="Donja kontura",
        ))

    # Vertikalne linije između kontura
    if rez.xx1 is not None and rez.xx2 is not None:
        for i in range(min(len(rez.xx1), len(rez.xx2))):
            fig.add_trace(go.Scatter3d(
                x=[rez.xx1[i], rez.xx2[i]],
                y=[rez.yy1[i], rez.yy2[i]],
                z=[rez.zz1[i], rez.zz2[i]],
                mode="lines",
                line=dict(color="#888", width=1),
                showlegend=False,
            ))

    fig.update_layout(
        title=f"Kupa: {rez.naziv}  (V={rez.zapremina/1e6:.2f} Mm³)",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z (m)",
            aspectmode="data",
        ),
        height=450,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
    )
    return fig


# ---------------------------------------------------------------------------
# Priprema ZIP za download
# ---------------------------------------------------------------------------

def napravi_zip(
    svi: list[RezultatTacke],
    validni: list[RezultatTacke],
    ts: str,
) -> bytes:
    """Pakuje Excel + DXF u ZIP bajt-stream za download."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:

        # Excel — svi rezultati
        if svi:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
                p = izvezi_excel(svi, tf.name)
            zf.write(p, arcname=f"{ts}_export_ga.csv")

        # Excel — validni
        if validni:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
                p = izvezi_excel(validni, tf.name)
            zf.write(p, arcname=f"{ts}_export_ga_final.csv")

        # DXF
        if validni:
            with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tf:
                dxf_path = Path(tf.name)
            with DxfWriter(dxf_path) as dxf:
                for j, rez in enumerate(validni, 1):
                    dxf.postavi(sloj=j, boja=(0.0, 0.0, 1.0))
                    if rez.xx1 is not None:
                        dxf.polilinija(rez.xx1, rez.yy1, rez.zz1)
                    if rez.xx2 is not None:
                        dxf.polilinija(rez.xx2, rez.yy2, rez.zz2)
            zf.write(dxf_path, arcname=f"{ts}_line_export_ga_24.dxf")

    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Streamlit aplikacija
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Optimizacija odlagališta",
    page_icon="⛏",
    layout="wide",
)

st.title("⛏ Optimizacija lokacije odlagališta")
st.caption("Python migracija MATLAB IzvrsniKodBuvac.m — genetski algoritam + 3D vizualizacija")

# Session state za čuvanje rezultata između reruna
if "podaci" not in st.session_state:
    st.session_state.podaci = None
if "rezultati" not in st.session_state:
    st.session_state.rezultati = []
if "validni" not in st.session_state:
    st.session_state.validni = []


# ---------------------------------------------------------------------------
# Sidebar — upload i parametri
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Ulazni podaci")

    with st.expander("📂 Fajlovi", expanded=True):
        f_teren   = st.file_uploader("Fajl terena (XYZ, CSV)", type=["txt", "csv"])
        f_zone    = st.file_uploader("Ekonomske zone", type=["txt"])
        f_cm      = st.file_uploader("Centar masa", type=["txt", "csv"])
        f_granice = st.file_uploader("Granica zone interesa", type=["txt", "csv"])
        f_params  = st.file_uploader("Dodatni parametri", type=["txt"])

    st.divider()
    st.header("Parametri GA")

    ugao = st.slider("Ugao kosine (°)", 15.0, 60.0, 37.0, 0.5)
    n_tacaka = st.slider("Monte Carlo tačaka", 10, 500, 100, 10)
    n_ponavljanja = st.slider("Ponavljanja", 1, 5, 1)
    min_v = st.number_input("Min zapremina (m³)", value=100_000.0, step=10_000.0, format="%.0f")
    max_v = st.number_input("Max zapremina (m³)", value=50_000_000.0, step=1_000_000.0, format="%.0f")
    populacija = st.slider("GA populacija", 10, 100, 30, 5)
    verzija = st.radio("Bounds verzija", ["buvac", "v1"], index=0)

    st.divider()
    pokreni = st.button("▶ Pokreni optimizaciju", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_teren, tab_optimizacija, tab_rezultati, tab_detalji = st.tabs([
    "🗺 Teren", "⚙ Optimizacija", "📊 Rezultati", "🔍 Detalji kupe"
])


# ---------------------------------------------------------------------------
# Tab 1: Teren — upload + prikaz
# ---------------------------------------------------------------------------
with tab_teren:
    if f_teren and f_granice:
        if st.button("Učitaj i prikaži teren"):
            with st.spinner("Učitavam teren..."):
                try:
                    p_teren   = spremi_upload(f_teren)
                    p_zone    = spremi_upload(f_zone) if f_zone else None
                    p_cm      = spremi_upload(f_cm) if f_cm else None
                    p_granice = spremi_upload(f_granice)
                    p_params  = spremi_upload(f_params) if f_params else None

                    teren = ucitaj_teren(p_teren)
                    granice = ucitaj_granice_zone(p_granice)

                    dobre_zone, lose_zone = (
                        ucitaj_ekonomske_zone(p_zone) if p_zone else ([], [])
                    )
                    centar_masa = (
                        ucitaj_centar_masa(p_cm) if p_cm
                        else np.array([
                            (granice.x_range[0]+granice.x_range[1])/2,
                            (granice.y_range[0]+granice.y_range[1])/2,
                            0.0,
                        ])
                    )
                    params = (
                        ucitaj_dodatne_parametre(p_params) if p_params
                        else DodatniParametri()
                    )

                    st.session_state.podaci = UlazniPodaci(
                        teren=teren, dobre_zone=dobre_zone, lose_zone=lose_zone,
                        centar_masa=centar_masa, granice=granice, parametri=params,
                    )
                    st.success(f"Teren učitan: {teren.vertices.shape[0]:,} tačaka, "
                               f"{teren.faces.shape[0]:,} trouglova")

                except Exception as e:
                    st.error(f"Greška pri učitavanju: {e}")

        if st.session_state.podaci:
            podaci = st.session_state.podaci
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Tačke terena", f"{podaci.teren.vertices.shape[0]:,}")
            col2.metric("Dobre zone", len(podaci.dobre_zone))
            col3.metric("Loše zone", len(podaci.lose_zone))
            col4.metric("mv (baza kupe)", f"{podaci.parametri.nadmorska_visina} m")

            if go:
                fig = plotly_teren_2d(podaci.teren.vertices, "Teren — prikaz po visini")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Instalirajte plotly za vizualizaciju: pip install plotly")
    else:
        st.info("👈 Upload fajl terena i granicu zone u sidebar-u da počneš.")


# ---------------------------------------------------------------------------
# Tab 2: Optimizacija — pokretanje
# ---------------------------------------------------------------------------
with tab_optimizacija:
    if pokreni:
        # Provjeri da li su podaci učitani
        if st.session_state.podaci is None:
            if not all([f_teren, f_granice]):
                st.error("Potrebno je uploadovati barem fajl terena i granice zone.")
                st.stop()
            # Učitaj automatski
            with st.spinner("Učitavam podatke..."):
                p_teren   = spremi_upload(f_teren)
                p_granice = spremi_upload(f_granice)
                p_zone    = spremi_upload(f_zone) if f_zone else None
                p_cm      = spremi_upload(f_cm) if f_cm else None
                p_params  = spremi_upload(f_params) if f_params else None

                teren   = ucitaj_teren(p_teren)
                granice = ucitaj_granice_zone(p_granice)
                dobre_zone, lose_zone = ucitaj_ekonomske_zone(p_zone) if p_zone else ([], [])
                centar_masa = (ucitaj_centar_masa(p_cm) if p_cm
                               else np.array([
                                   (granice.x_range[0]+granice.x_range[1])/2,
                                   (granice.y_range[0]+granice.y_range[1])/2, 0.0]))
                params = (ucitaj_dodatne_parametre(p_params) if p_params
                          else DodatniParametri())
                st.session_state.podaci = UlazniPodaci(
                    teren=teren, dobre_zone=dobre_zone, lose_zone=lose_zone,
                    centar_masa=centar_masa, granice=granice, parametri=params,
                )

        podaci = st.session_state.podaci
        granice = podaci.granice

        ctx = GAKontekst(
            centar_masa=podaci.centar_masa,
            ugao=ugao,
            mnv=podaci.parametri.nadmorska_visina,
            donja_granica_zapremine=min_v,
            gornja_granica_zapremine=max_v,
            uslov_distance=podaci.parametri.uslov_distance,
            teren=podaci.teren,
            zona_x=granice.x_poly,
            zona_y=granice.y_poly,
            dobre_zone=podaci.dobre_zone,
        )
        opcije = GAOpcije(
            populacija=populacija,
            max_generacija=podaci.parametri.broj_generacija,
        )

        # ---- Faza 1: Monte Carlo ----
        with st.status("🎲 Monte Carlo generisanje tačaka...", expanded=True) as status:
            sve_dobre = []
            for rep in range(1, n_ponavljanja + 1):
                st.write(f"Ponavljanje {rep}/{n_ponavljanja}...")
                tacke = generiši_tačke(
                    n=n_tacaka,
                    x_range=granice.x_range,
                    y_range=granice.y_range,
                    z_range=granice.z_range,
                    zona_x=granice.x_poly,
                    zona_y=granice.y_poly,
                    lose_zone=podaci.lose_zone,
                )
                # Provjera geometrije kupe
                for j, tacka in enumerate(tacke):
                    wx, wy, wz = float(tacka[0]), float(tacka[1]), float(tacka[2])
                    rez = zapremina_kupe(wx, wy, wz, ugao, 0.001,
                                        podaci.parametri.nadmorska_visina,
                                        podaci.teren, granice.x_poly, granice.y_poly)
                    if (rez.zapremina < 40_000_000 and
                            rez.intersect_surface is not None and
                            rez.intersect_surface.vertices.shape[0] > 0):
                        sve_dobre.append((f"p_{j+1}_{rep}", wx, wy, wz))

            status.update(
                label=f"✅ Monte Carlo završen — {len(sve_dobre)} dobrih tačaka",
                state="complete",
            )

        if not sve_dobre:
            st.warning("Nema dobrih tačaka. Povećaj broj tačaka ili provjeri parametre.")
            st.stop()

        # ---- Faza 2: GA ----
        tacke_array = np.array([[wx, wy, wz] for _, wx, wy, wz in sve_dobre])
        svi_rez: list[RezultatTacke] = []
        validni_rez: list[RezultatTacke] = []

        progress = st.progress(0, text="Pokretanje GA...")
        n = len(tacke_array)

        for i, (naziv, wx, wy, wz) in enumerate(sve_dobre):
            progress.progress((i + 1) / n,
                               text=f"GA tačka {i+1}/{n}: ({wx:.0f}, {wy:.0f})")
            rez = optimizuj_tacku(naziv, wx, wy, ctx, opcije, verzija)
            if rez:
                svi_rez.append(rez)
                if rez.unutar_zone:
                    validni_rez.append(rez)

        progress.progress(1.0, text="GA završen ✅")

        st.session_state.rezultati = svi_rez
        st.session_state.validni   = validni_rez

        # Kratki sažetak
        col1, col2, col3 = st.columns(3)
        col1.metric("Monte Carlo dobrih", len(sve_dobre))
        col2.metric("GA rezultata", len(svi_rez))
        col3.metric("Validnih (final)", len(validni_rez))

        if validni_rez:
            best = min(validni_rez, key=lambda r: r.f_vrednost)
            st.success(f"Najbolja lokacija: **{best.naziv}** — "
                       f"f={best.f_vrednost:.4f}, V={best.zapremina/1e6:.2f} Mm³, "
                       f"zona: {best.zone or '—'}")
    else:
        st.info("Podesi parametre u sidebar-u i pritisni **▶ Pokreni optimizaciju**.")


# ---------------------------------------------------------------------------
# Tab 3: Rezultati — tabela + mapa
# ---------------------------------------------------------------------------
with tab_rezultati:
    svi_rez   = st.session_state.rezultati
    validni_rez = st.session_state.validni

    if not svi_rez:
        st.info("Pokreni optimizaciju da vidiš rezultate.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Ukupno rezultata", len(svi_rez))
        col2.metric("Validnih", len(validni_rez))

        if validni_rez:
            best = min(validni_rez, key=lambda r: r.f_vrednost)
            col3.metric("Najmanji f", f"{best.f_vrednost:.4f}")
            col4.metric("Najveća zapremina", f"{max(r.zapremina for r in validni_rez)/1e6:.2f} Mm³")

        # Mapa
        if go and st.session_state.podaci:
            podaci = st.session_state.podaci
            fig = plotly_rezultati(
                validni_rez,
                podaci.teren.vertices,
                podaci.granice.x_poly,
                podaci.granice.y_poly,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Tabela
        try:
            import pandas as pd
            redovi = [r.kao_red() for r in svi_rez]
            df = pd.DataFrame(redovi, columns=RezultatTacke.ZAGLAVLJE)
            for kol in ["X_koordinata", "Y_koordinata", "Z_koordinata", "K",
                        "Funkcija_cilja", "Zapremina", "distanca", "c1", "c2", "c3"]:
                if kol in df.columns:
                    df[kol] = pd.to_numeric(df[kol], errors="coerce")
            st.dataframe(
                df.style.background_gradient(subset=["Funkcija_cilja"], cmap="RdYlGn_r"),
                use_container_width=True,
                height=400,
            )
        except ImportError:
            # Bez pandas — jednostavna tabela
            zaglavlje_str = " | ".join(RezultatTacke.ZAGLAVLJE)
            st.text(zaglavlje_str)
            for r in svi_rez:
                st.text(" | ".join(str(v) for v in r.kao_red()))

        # Download
        if svi_rez:
            ts = timestamp_string()
            zip_bytes = napravi_zip(svi_rez, validni_rez, ts)
            st.download_button(
                label="⬇ Preuzmi rezultate (CSV + DXF u ZIP)",
                data=zip_bytes,
                file_name=f"{ts}_rezultati.zip",
                mime="application/zip",
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# Tab 4: Detalji jedne kupe
# ---------------------------------------------------------------------------
with tab_detalji:
    if not st.session_state.validni:
        st.info("Nema validnih rezultata. Pokreni optimizaciju prvo.")
    else:
        validni_rez = st.session_state.validni
        sortirani = sorted(validni_rez, key=lambda r: r.f_vrednost)
        opcije_select = {
            f"{r.naziv} (f={r.f_vrednost:.4f})": r for r in sortirani
        }
        odabir = st.selectbox("Odaberi tačku:", list(opcije_select.keys()))
        rez = opcije_select[odabir]

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Funkcija cilja f", f"{rez.f_vrednost:.4f}")
        col2.metric("Zapremina", f"{rez.zapremina/1e6:.3f} Mm³")
        col3.metric("Distanca od CM", f"{rez.distanca:.0f} m")
        col4.metric("Visina vrha wz", f"{rez.wz:.1f} m")
        col5.metric("Širina k", f"{rez.k:.1f} m")

        col6, col7, col8 = st.columns(3)
        col6.metric("c1 (transport)", f"{rez.c1:,.0f}")
        col7.metric("c2 (iskopavanje)", f"{rez.c2:,.0f}")
        col8.metric("c3 (zemljište)", f"{rez.c3:,.0f}")

        st.caption(f"Zone: {rez.zone or '—'}")

        if go and rez.xx1 is not None:
            fig3d = plotly_kupa_3d(rez)
            st.plotly_chart(fig3d, use_container_width=True)
        else:
            st.info("Instaliraj plotly za 3D prikaz kupe.")
