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


def plotly_teren_3d(
    teren_vertices: np.ndarray,
    teren_faces: np.ndarray,
    centar_masa: np.ndarray = None,
    zona_x: np.ndarray = None,
    zona_y: np.ndarray = None,
    naslov: str = "3D prikaz terena",
) -> go.Figure:
    """Interaktivni 3D prikaz terena kao triangulisane površine.

    Teren — Mesh3d bojana po visini (zelena→smeđa→pijesak)
    Centar masa — crvena zvjezdica
    Granica zone — žuta linija na visini terena
    """
    fig = go.Figure()

    # Decimiraj za prikaz ako je previše tačaka (browser limit)
    v = teren_vertices
    f = teren_faces
    if len(v) > 20000:
        # Uzimamo svaku N-tu tačku i samo trouglove koji koriste te tačke
        korak = max(2, len(v) // 15000)
        maska = np.zeros(len(v), dtype=bool)
        maska[::korak] = True
        idx_mapa = np.full(len(v), -1, dtype=int)
        idx_mapa[maska] = np.arange(maska.sum())
        v = teren_vertices[maska]
        maska_f = maska[f[:, 0]] & maska[f[:, 1]] & maska[f[:, 2]]
        f = idx_mapa[teren_faces[maska_f]]

    z_min = v[:, 2].min()
    z_max = v[:, 2].max()
    z_range = max(z_max - z_min, 1.0)
    intensity = (v[:, 2] - z_min) / z_range

    # Teren — Mesh3d
    fig.add_trace(go.Mesh3d(
        x=v[:, 0],
        y=v[:, 1],
        z=v[:, 2],
        i=f[:, 0],
        j=f[:, 1],
        k=f[:, 2],
        intensity=intensity,
        colorscale=[
            [0.00, "#1a4a0a"],   # tamno zelena — najniža tačka
            [0.25, "#3a7a1a"],   # zelena
            [0.50, "#7a6040"],   # smeđa — sredina
            [0.75, "#a08060"],   # svjetla smeđa
            [1.00, "#d4c4a0"],   # pijesak — vrh
        ],
        showscale=True,
        colorbar=dict(
            title=dict(text="Visina (m)", side="right"),
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            ticktext=[
                f"{z_min:.0f}m",
                f"{z_min + z_range*0.25:.0f}m",
                f"{z_min + z_range*0.50:.0f}m",
                f"{z_min + z_range*0.75:.0f}m",
                f"{z_max:.0f}m",
            ],
            len=0.6,
        ),
        opacity=1.0,
        flatshading=False,
        lighting=dict(
            ambient=0.6,
            diffuse=0.8,
            specular=0.2,
            roughness=0.8,
        ),
        lightposition=dict(x=1, y=1, z=2),
        name="Teren",
        hovertemplate="X: %{x:.0f}<br>Y: %{y:.0f}<br>Z: %{z:.1f} m<extra>Teren</extra>",
    ))

    # Granica zone — linija na visini terena
    if zona_x is not None and zona_y is not None and len(zona_x) >= 2:
        # Interpoliraj Z iz terena za svaku tačku granice
        from scipy.spatial import cKDTree
        tree = cKDTree(teren_vertices[:, :2])
        _, idx = tree.query(np.column_stack([zona_x, zona_y]))
        zona_z = teren_vertices[idx, 2] + 5.0   # malo iznad terena da se vidi

        fig.add_trace(go.Scatter3d(
            x=np.append(zona_x, zona_x[0]),
            y=np.append(zona_y, zona_y[0]),
            z=np.append(zona_z, zona_z[0]),
            mode="lines",
            line=dict(color="#FFD700", width=5),
            name="Granica zone interesa",
            hovertemplate="Granica interesne zone<extra></extra>",
        ))

    # Centar masa — crvena zvjezdica na terenu
    if centar_masa is not None:
        from scipy.spatial import cKDTree
        tree = cKDTree(teren_vertices[:, :2])
        _, idx_cm = tree.query([[float(centar_masa[0]), float(centar_masa[1])]])
        cm_z = float(teren_vertices[idx_cm[0], 2]) + 10.0

        fig.add_trace(go.Scatter3d(
            x=[float(centar_masa[0])],
            y=[float(centar_masa[1])],
            z=[cm_z],
            mode="markers+text",
            marker=dict(size=10, color="#FF00FF", symbol="diamond",
                        line=dict(width=2, color="white")),
            text=["Centar masa"],
            textposition="top center",
            textfont=dict(color="#FF00FF", size=13),
            name="Centar masa",
            hovertemplate=(
                f"<b>Centar masa</b><br>"
                f"X: {float(centar_masa[0]):.0f}<br>"
                f"Y: {float(centar_masa[1]):.0f}<br>"
                f"Z terena: {cm_z:.1f} m<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(text=naslov, font=dict(size=15)),
        scene=dict(
            xaxis_title="X koordinata",
            yaxis_title="Y koordinata",
            zaxis_title="Visina (m)",
            aspectmode="data",
            bgcolor="rgba(5,10,20,1)",
            xaxis=dict(
                showbackground=True,
                backgroundcolor="rgba(5,10,20,1)",
                gridcolor="#223",
                title_font=dict(size=11),
            ),
            yaxis=dict(
                showbackground=True,
                backgroundcolor="rgba(5,10,20,1)",
                gridcolor="#223",
                title_font=dict(size=11),
            ),
            zaxis=dict(
                showbackground=True,
                backgroundcolor="rgba(5,10,20,1)",
                gridcolor="#223",
                title_font=dict(size=11),
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        height=650,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            bgcolor="rgba(20,20,30,0.8)",
            font=dict(size=11),
        ),
    )
    return fig


def plotly_rezultati(
    rezultati: list[RezultatTacke],
    teren_vertices: np.ndarray,
    zona_x: np.ndarray,
    zona_y: np.ndarray,
    dobre_zone: list = None,
    lose_zone: list = None,
    centar_masa: np.ndarray = None,
) -> go.Figure:
    """Interaktivna 2D mapa sa terenom, ekonomskim zonama, granicom i rezultatima."""

    BOJE_ZONA = {
        "Z-1": "#378ADD",
        "Z-3": "#1D9E75",
        "Z-4": "#BA7517",
        "Z-5": "#E24B4A",
    }

    def boja_zone(naziv):
        for prefiks, boja in BOJE_ZONA.items():
            if naziv.startswith(prefiks):
                return boja
        return "#888888"

    fig = go.Figure()

    # Teren u pozadini
    korak = max(1, len(teren_vertices) // 3000)
    t = teren_vertices[::korak]
    fig.add_trace(go.Scattergl(
        x=t[:, 0], y=t[:, 1],
        mode="markers",
        marker=dict(size=2, color=t[:, 2], colorscale="Greys", opacity=0.35, showscale=False),
        name="Teren",
        showlegend=True,
        hovertemplate="X: %{x:.0f}<br>Y: %{y:.0f}<br>Z: %{customdata:.1f}m",
        customdata=t[:, 2],
    ))

    # Ekonomske zone (dobre)
    if dobre_zone:
        dodane_u_legendu = set()
        for zona in dobre_zone:
            boja = boja_zone(zona.naziv)
            tip = next((p for p in BOJE_ZONA if zona.naziv.startswith(p)), "Ostalo")
            u_legendi = tip not in dodane_u_legendu
            if u_legendi:
                dodane_u_legendu.add(tip)
            x_z = np.append(zona.x_data, zona.x_data[0])
            y_z = np.append(zona.y_data, zona.y_data[0])
            fig.add_trace(go.Scatter(
                x=x_z, y=y_z,
                mode="lines",
                fill="toself",
                fillcolor=boja,
                opacity=0.25,
                line=dict(color=boja, width=1),
                name=f"Zona {tip}",
                legendgroup=tip,
                showlegend=u_legendi,
                hovertemplate=f"<b>{zona.naziv}</b><br>Površina: {zona.povrsina:,.0f} m²",
            ))

    # Loše zone (Z-5)
    if lose_zone:
        dodana_losa = False
        for zona in lose_zone:
            x_z = np.append(zona.x_data, zona.x_data[0])
            y_z = np.append(zona.y_data, zona.y_data[0])
            fig.add_trace(go.Scatter(
                x=x_z, y=y_z,
                mode="lines",
                fill="toself",
                fillcolor="#E24B4A",
                opacity=0.3,
                line=dict(color="#E24B4A", width=1.5, dash="dot"),
                name="Zona Z-5 (zabranjena)",
                legendgroup="Z-5",
                showlegend=not dodana_losa,
                hovertemplate=f"<b>{zona.naziv}</b> — zabranjena zona",
            ))
            dodana_losa = True

    # Granica interesne zone
    fig.add_trace(go.Scatter(
        x=np.append(zona_x, zona_x[0]),
        y=np.append(zona_y, zona_y[0]),
        mode="lines",
        line=dict(color="#FFD700", width=3, dash="dash"),
        name="Granica zone interesa",
        hovertemplate="Granica interesne zone",
    ))

    # Centar masa — zvjezdica
    if centar_masa is not None:
        fig.add_trace(go.Scatter(
            x=[float(centar_masa[0])],
            y=[float(centar_masa[1])],
            mode="markers+text",
            marker=dict(symbol="star", size=22, color="#FF00FF",
                        line=dict(width=2, color="white")),
            text=["  Centar masa"],
            textposition="middle right",
            textfont=dict(size=13, color="#FF00FF"),
            name="Centar masa",
            hovertemplate=(
                f"<b>Centar masa</b><br>"
                f"X: {float(centar_masa[0]):.0f}<br>"
                f"Y: {float(centar_masa[1]):.0f}<extra></extra>"
            ),
        ))

    # Rezultati GA
    if rezultati:
        validni   = [r for r in rezultati if r.unutar_zone]
        nevalidni = [r for r in rezultati if not r.unutar_zone]

        if validni:
            fig.add_trace(go.Scatter(
                x=[r.wx for r in validni],
                y=[r.wy for r in validni],
                mode="markers",
                marker=dict(
                    symbol="circle",
                    size=14,
                    color=[r.f_vrednost for r in validni],
                    colorscale="RdYlGn_r",
                    showscale=True,
                    colorbar=dict(title="f vrijednost", x=1.02),
                    line=dict(width=2, color="white"),
                ),
                name="Validne lokacije",
                customdata=[[r.naziv, r.f_vrednost, r.zapremina/1e6, r.wz, r.k, r.zone]
                            for r in validni],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "f = %{customdata[1]:.4f}<br>"
                    "V = %{customdata[2]:.2f} Mm³<br>"
                    "wz = %{customdata[3]:.1f} m<br>"
                    "k = %{customdata[4]:.1f} m<br>"
                    "Zone: %{customdata[5]}<extra></extra>"
                ),
            ))

        if nevalidni:
            fig.add_trace(go.Scatter(
                x=[r.wx for r in nevalidni],
                y=[r.wy for r in nevalidni],
                mode="markers",
                marker=dict(symbol="x", size=10, color="#E24B4A",
                            line=dict(width=2, color="#E24B4A")),
                name="Van zone (odbačene)",
                text=[r.naziv for r in nevalidni],
                hovertemplate="<b>%{text}</b> — van zone<extra></extra>",
            ))

        # Najbolja lokacija — zlatna zvjezdica
        if validni:
            best = min(validni, key=lambda r: r.f_vrednost)
            fig.add_trace(go.Scatter(
                x=[best.wx], y=[best.wy],
                mode="markers+text",
                marker=dict(symbol="star", size=26, color="#FFD700",
                            line=dict(width=2, color="black")),
                text=[f"  BEST: {best.naziv}"],
                textposition="middle right",
                textfont=dict(size=12, color="#FFD700"),
                name=f"Najbolja lokacija",
                hovertemplate=(
                    f"<b>Najbolja lokacija</b><br>"
                    f"f = {best.f_vrednost:.4f}<br>"
                    f"V = {best.zapremina/1e6:.2f} Mm³<extra></extra>"
                ),
            ))

    fig.update_layout(
        title="Mapa optimizacije odlagališta",
        xaxis_title="X koordinata",
        yaxis_title="Y koordinata",
        height=620,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(30,30,30,0.7)",
            font=dict(size=11),
        ),
        margin=dict(l=0, r=80, t=40, b=0),
        plot_bgcolor="rgba(15,15,25,0.95)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plotly_kupa_3d(
    rez: RezultatTacke,
    teren_vertices: np.ndarray = None,
    teren_faces: np.ndarray = None,
) -> go.Figure:
    """3D prikaz kupe i terena ispod nje.

    Teren — smeđe-zelena površina (Plotly Mesh3d)
    Kupa  — gornja kontura zelena, donja plava, bočne linije sive
    """
    fig = go.Figure()

    # ---- Teren ----
    if teren_vertices is not None and teren_faces is not None:
        # Decimiramo teren oko kupe — uzimamo tačke u radijusu od kupe
        if rez.xx1 is not None:
            cx = float(rez.wx)
            cy = float(rez.wy)
            # Radijus prikaza = 3× širina kupe (k) + mali buffer
            r_prikaz = float(rez.k) * 4.0 + 500.0
            dist = np.sqrt((teren_vertices[:, 0] - cx)**2 +
                           (teren_vertices[:, 1] - cy)**2)
            maska_v = dist < r_prikaz
            # Preslikaj indekse (samo trouglovi gdje su sva 3 vrha unutar maske)
            idx_mapa = np.full(len(teren_vertices), -1, dtype=int)
            idx_mapa[maska_v] = np.arange(maska_v.sum())
            v_lok = teren_vertices[maska_v]

            maska_f = (
                maska_v[teren_faces[:, 0]] &
                maska_v[teren_faces[:, 1]] &
                maska_v[teren_faces[:, 2]]
            )
            f_lok = teren_faces[maska_f]
            f_lok = idx_mapa[f_lok]
        else:
            korak = max(1, len(teren_vertices) // 5000)
            v_lok = teren_vertices[::korak]
            f_lok = None

        if len(v_lok) >= 3:
            z_min = v_lok[:, 2].min()
            z_max = v_lok[:, 2].max()
            z_norm = (v_lok[:, 2] - z_min) / max(z_max - z_min, 1)

            mesh_kwargs = dict(
                x=v_lok[:, 0],
                y=v_lok[:, 1],
                z=v_lok[:, 2],
                intensity=z_norm,
                colorscale=[
                    [0.0,  "#2d5a1b"],   # tamno zelena — nizina
                    [0.3,  "#5a8c3c"],   # srednje zelena
                    [0.6,  "#8B7355"],   # smeđa — srednja visina
                    [0.85, "#A0896B"],   # svjetlo smeđa
                    [1.0,  "#C8B89A"],   # pijesak — vrh
                ],
                showscale=False,
                opacity=0.85,
                name="Teren",
                hovertemplate="X: %{x:.0f}<br>Y: %{y:.0f}<br>Z: %{z:.1f} m<extra>Teren</extra>",
            )
            if f_lok is not None and len(f_lok) > 0:
                mesh_kwargs["i"] = f_lok[:, 0]
                mesh_kwargs["j"] = f_lok[:, 1]
                mesh_kwargs["k"] = f_lok[:, 2]
            else:
                # Bez lica — samo oblak tačaka kao površina
                from scipy.spatial import Delaunay as _D
                try:
                    tri = _D(v_lok[:, :2])
                    mesh_kwargs["i"] = tri.simplices[:, 0]
                    mesh_kwargs["j"] = tri.simplices[:, 1]
                    mesh_kwargs["k"] = tri.simplices[:, 2]
                except Exception:
                    pass

            fig.add_trace(go.Mesh3d(**mesh_kwargs))

    # ---- Površina kupe (ispunjena) ----
    if rez.xx1 is not None and rez.xx2 is not None:
        # Konstruiši mrežu kupe od 9 tačaka gornje + 9 donje konture
        n_k = len(rez.xx1)
        x_kupa = np.concatenate([rez.xx1, rez.xx2])
        y_kupa = np.concatenate([rez.yy1, rez.yy2])
        z_kupa = np.concatenate([rez.zz1, rez.zz2])

        # Lica kupe — trapezoidni panel između gornje i donje
        i_lica, j_lica, k_lica = [], [], []
        for idx in range(n_k - 1):
            # Gornji trougao panela
            i_lica.append(idx);          j_lica.append(idx + 1);          k_lica.append(idx + n_k)
            # Donji trougao panela
            i_lica.append(idx + 1);     j_lica.append(idx + n_k + 1);    k_lica.append(idx + n_k)

        fig.add_trace(go.Mesh3d(
            x=x_kupa, y=y_kupa, z=z_kupa,
            i=i_lica, j=j_lica, k=k_lica,
            color="#FF8C00",
            opacity=0.45,
            name="Kupa (površina)",
            showscale=False,
            hovertemplate=f"<b>Kupa: {rez.naziv}</b><br>Z: %{{z:.1f}} m<extra></extra>",
        ))

    # ---- Gornja kontura kupe ----
    if rez.xx1 is not None:
        fig.add_trace(go.Scatter3d(
            x=np.append(rez.xx1, rez.xx1[0]),
            y=np.append(rez.yy1, rez.yy1[0]),
            z=np.append(rez.zz1, rez.zz1[0]),
            mode="lines",
            line=dict(color="#00FF88", width=5),
            name="Gornja kontura kupe",
        ))

    # ---- Donja kontura kupe ----
    if rez.xx2 is not None:
        fig.add_trace(go.Scatter3d(
            x=np.append(rez.xx2, rez.xx2[0]),
            y=np.append(rez.yy2, rez.yy2[0]),
            z=np.append(rez.zz2, rez.zz2[0]),
            mode="lines",
            line=dict(color="#00AAFF", width=5),
            name="Donja kontura kupe",
        ))

    # ---- Bočne linije kupe (gore→dolje) ----
    if rez.xx1 is not None and rez.xx2 is not None:
        n_k = min(len(rez.xx1), len(rez.xx2))
        for i in range(n_k):
            fig.add_trace(go.Scatter3d(
                x=[rez.xx1[i], rez.xx2[i]],
                y=[rez.yy1[i], rez.yy2[i]],
                z=[rez.zz1[i], rez.zz2[i]],
                mode="lines",
                line=dict(color="#FFAA00", width=2),
                showlegend=False,
            ))

    # ---- Vrh kupe (centar gornje konture) ----
    fig.add_trace(go.Scatter3d(
        x=[rez.wx], y=[rez.wy], z=[rez.wz],
        mode="markers+text",
        marker=dict(size=8, color="#FF0000", symbol="diamond"),
        text=[f"wz={rez.wz:.1f}m"],
        textposition="top center",
        textfont=dict(color="#FF4444", size=12),
        name="Vrh kupe",
    ))

    fig.update_layout(
        title=dict(
            text=f"3D prikaz kupe: {rez.naziv} — V={rez.zapremina/1e6:.2f} Mm³ | wz={rez.wz:.1f}m | k={rez.k:.1f}m",
            font=dict(size=14),
        ),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Visina (m)",
            aspectmode="data",
            bgcolor="rgba(10,10,20,1)",
            xaxis=dict(gridcolor="#333", showbackground=True, backgroundcolor="rgba(10,10,20,1)"),
            yaxis=dict(gridcolor="#333", showbackground=True, backgroundcolor="rgba(10,10,20,1)"),
            zaxis=dict(gridcolor="#333", showbackground=True, backgroundcolor="rgba(10,10,20,1)"),
        ),
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(20,20,30,0.8)",
            font=dict(size=11),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
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
    min_v = st.number_input("Min zapremina (m³)", value=100_000.0, step=10_000.0, format="%.0f")
    max_v = st.number_input("Max zapremina (m³)", value=50_000_000.0, step=1_000_000.0, format="%.0f")
    populacija = st.slider("GA populacija", 10, 100, 30, 5)
    verzija = st.radio(
        "Bounds verzija",
        ["auto", "buvac", "v1"],
        index=0,
        help="auto = iz terena (preporučeno) | buvac = originalni Buvac | v1 = starija verzija",
    )

    st.divider()
    st.header("Način rada")
    nacin_rada = st.radio(
        "Odaberi način generisanja tačaka:",
        ["🎲 Monte Carlo", "📍 Konkretna tačka"],
        index=0,
    )

    if nacin_rada == "🎲 Monte Carlo":
        n_tacaka = st.slider("Broj nasumičnih tačaka", 10, 500, 100, 10)
        n_ponavljanja = st.slider("Ponavljanja", 1, 5, 1)
        # Konkretna tačka nije aktivna
        konk_wx = None
        konk_wy = None
        konk_wz = None
    else:
        st.caption("Unesi koordinate tačke na kojoj hoćeš da optimizuješ kupu.")
        # Default koordinate — centar učitanog terena/zone, ne Buvac
        if st.session_state.podaci is not None:
            _g = st.session_state.podaci.granice
            _cm = st.session_state.podaci.centar_masa
            _def_x = float(_cm[0]) if _cm is not None else (_g.x_range[0] + _g.x_range[1]) / 2
            _def_y = float(_cm[1]) if _cm is not None else (_g.y_range[0] + _g.y_range[1]) / 2
            _def_z = float(st.session_state.podaci.parametri.nadmorska_visina or
                          st.session_state.podaci.teren.vertices[:, 2].mean())
        else:
            _def_x, _def_y, _def_z = 0.0, 0.0, 0.0
        konk_wx = st.number_input("X koordinata", value=_def_x, format="%.2f")
        konk_wy = st.number_input("Y koordinata", value=_def_y, format="%.2f")
        konk_wz = st.number_input("Z koordinata (visina)", value=_def_z, format="%.2f",
                                   help="Početna visina — GA će optimizovati wz unutar bounds-a")
        n_tacaka = 1
        n_ponavljanja = 1

    st.divider()
    pokreni = st.button("▶ Pokreni optimizaciju", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_teren, tab_optimizacija, tab_rezultati, tab_detalji, tab_frustum = st.tabs([
    "🗺 Teren", "⚙ Optimizacija", "📊 Rezultati", "🔍 Detalji kupe", "🔺 Frustum"
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
                st.caption(
                    "💡 Rotiraj mišem (lijevo dugme), zumiraj (scroll), "
                    "pomjeri (desno dugme). Hover mišem za koordinate i visinu."
                )
                fig3d_teren = plotly_teren_3d(
                    teren_vertices=podaci.teren.vertices,
                    teren_faces=podaci.teren.faces,
                    centar_masa=podaci.centar_masa,
                    zona_x=podaci.granice.x_poly,
                    zona_y=podaci.granice.y_poly,
                    naslov=(
                        f"3D teren — {podaci.teren.vertices.shape[0]:,} tačaka | "
                        f"Z: {podaci.teren.vertices[:,2].min():.0f}m – "
                        f"{podaci.teren.vertices[:,2].max():.0f}m"
                    ),
                )
                st.plotly_chart(fig3d_teren, use_container_width=True)
            else:
                st.info("Instalirajte plotly za vizualizaciju: pip install plotly")
    else:
        st.info("👈 Upload fajl terena i granicu zone u sidebar-u da počneš.")


# ---------------------------------------------------------------------------
# Tab 2: Optimizacija — pokretanje
# ---------------------------------------------------------------------------
with tab_optimizacija:

    # Prikaz odabranog načina rada
    if nacin_rada == "📍 Konkretna tačka":
        st.info(
            f"**Konkretna tačka:** X = {konk_wx:.2f},  Y = {konk_wy:.2f},  Z = {konk_wz:.2f}  "
            f"— GA će optimizovati visinu (wz) i širinu (k) kupe na ovoj poziciji."
        )
    else:
        st.info(f"**Monte Carlo:** {n_tacaka} nasumičnih tačaka × {n_ponavljanja} ponavljanja")

    if pokreni:
        # --- Učitaj podatke ako nisu već ---
        if st.session_state.podaci is None:
            if not all([f_teren, f_granice]):
                st.error("Potrebno je uploadovati barem fajl terena i granice zone.")
                st.stop()
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

        svi_rez: list[RezultatTacke] = []
        validni_rez: list[RezultatTacke] = []

        # ================================================================
        # NAČIN 1: Konkretna tačka
        # ================================================================
        if nacin_rada == "📍 Konkretna tačka":
            with st.status("📍 Pokretanje GA za konkretnu tačku...", expanded=True) as status:
                st.write(f"Tačka: X={konk_wx:.2f}, Y={konk_wy:.2f}, Z={konk_wz:.2f}")
                st.write("Optimizujem wz i k...")

                rez = optimizuj_tacku(
                    naziv="konkretna_tacka",
                    wx=konk_wx,
                    wy=konk_wy,
                    ctx=ctx,
                    opcije=opcije,
                    verzija=verzija,
                )

                if rez:
                    svi_rez.append(rez)
                    if rez.unutar_zone:
                        validni_rez.append(rez)
                    status.update(label="✅ GA završen", state="complete")
                else:
                    status.update(
                        label="⚠ GA nije našao rješenje za ovu tačku",
                        state="error"
                    )
                    st.warning(
                        "GA nije uspio pronaći validnu kupu na ovoj poziciji. "
                        "Provjeri koordinate — tačka možda nije unutar zone interesa, "
                        "ili zapremina ne zadovoljava granice."
                    )

            # Direktni prikaz rezultata za konkretnu tačku
            if rez:
                st.subheader("Rezultat optimizacije")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("f vrijednost", f"{rez.f_vrednost:.4f}")
                c2.metric("Zapremina", f"{rez.zapremina/1e6:.3f} Mm³")
                c3.metric("wz (visina vrha)", f"{rez.wz:.1f} m")
                c4.metric("k (širina)", f"{rez.k:.1f} m")
                c5.metric("Distanca od CM", f"{rez.distanca:.0f} m")

                c6, c7, c8, c9 = st.columns(4)
                c6.metric("c1 transport", f"{rez.c1:,.0f}")
                c7.metric("c2 iskopavanje", f"{rez.c2:,.0f}")
                c8.metric("c3 zemljište", f"{rez.c3:,.0f}")
                c9.metric("Unutar zone", "✅ Da" if rez.unutar_zone else "❌ Ne")

                if rez.zone:
                    st.caption(f"Zone koje kupa pokriva: **{rez.zone}**")

        # ================================================================
        # NAČIN 2: Monte Carlo
        # ================================================================
        else:
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
                    for j, tacka in enumerate(tacke):
                        wx, wy, wz = float(tacka[0]), float(tacka[1]), float(tacka[2])
                        # k_test = 5% dijagonale zone (realna minimalna vrijednost)
                        _dx = granice.x_range[1] - granice.x_range[0]
                        _dy = granice.y_range[1] - granice.y_range[0]
                        import numpy as _npk
                        k_test = max(10.0, float(_npk.sqrt(_dx**2 + _dy**2)) * 0.03)
                        rez_k = zapremina_kupe(wx, wy, wz, ugao, k_test,
                                              podaci.parametri.nadmorska_visina,
                                              podaci.teren, granice.x_poly, granice.y_poly)
                        if (rez_k.zapremina < 40_000_000 and
                                rez_k.intersect_surface is not None and
                                rez_k.intersect_surface.vertices.shape[0] > 0):
                            sve_dobre.append((f"p_{j+1}_{rep}", wx, wy, wz))

                status.update(
                    label=f"✅ Monte Carlo — {len(sve_dobre)} dobrih tačaka",
                    state="complete",
                )

            if not sve_dobre:
                st.warning("Nema dobrih tačaka. Povećaj broj tačaka ili provjeri parametre.")
                st.stop()

            progress = st.progress(0, text="Pokretanje GA...")
            n_t = len(sve_dobre)
            for i, (naziv, wx, wy, wz) in enumerate(sve_dobre):
                progress.progress((i + 1) / n_t,
                                   text=f"GA tačka {i+1}/{n_t}: ({wx:.0f}, {wy:.0f})")
                rez = optimizuj_tacku(naziv, wx, wy, ctx, opcije, verzija)
                if rez:
                    svi_rez.append(rez)
                    if rez.unutar_zone:
                        validni_rez.append(rez)
            progress.progress(1.0, text="GA završen ✅")

        # --- Sažetak i čuvanje ---
        st.session_state.rezultati = svi_rez
        st.session_state.validni   = validni_rez

        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("GA rezultata ukupno", len(svi_rez))
        col2.metric("Validnih (unutar zone)", len(validni_rez))
        if validni_rez:
            best = min(validni_rez, key=lambda r: r.f_vrednost)
            col3.metric("Najmanji f", f"{best.f_vrednost:.4f}")
            st.success(
                f"Najbolja lokacija: **{best.naziv}** — "
                f"f={best.f_vrednost:.4f}, V={best.zapremina/1e6:.2f} Mm³, "
                f"zona: {best.zone or '—'}"
            )
        st.info("Detalji i mapa dostupni u tabovima **📊 Rezultati** i **🔍 Detalji kupe**.")

    else:
        if nacin_rada == "📍 Konkretna tačka":
            st.info(
                "Unesi koordinate tačke u sidebar-u i pritisni **▶ Pokreni optimizaciju**. "
                "GA će pronaći optimalnu visinu i širinu kupe tačno na toj poziciji."
            )
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
                dobre_zone=podaci.dobre_zone,
                lose_zone=podaci.lose_zone,
                centar_masa=podaci.centar_masa,
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
            teren_v = st.session_state.podaci.teren.vertices if st.session_state.podaci else None
            teren_f = st.session_state.podaci.teren.faces if st.session_state.podaci else None
            fig3d = plotly_kupa_3d(rez, teren_vertices=teren_v, teren_faces=teren_f)
            st.plotly_chart(fig3d, use_container_width=True)
        else:
            st.info("Instaliraj plotly za 3D prikaz kupe.")


# ---------------------------------------------------------------------------
# Tab 5: Frustum — direktni prikaz bez Monte Carlo / GA
# ---------------------------------------------------------------------------

def _frustum_plotly_ravan(
    cx: float, cy: float, z_top: float,
    r_top: float, alpha_deg: float,
    terrain_z: float = 0.0,
    depth_below: float = 10.0,
) -> tuple[go.Figure, dict]:
    """Plotly 3D prikaz frustuma na RAVNOM terenu."""

    alpha_rad  = np.radians(alpha_deg)
    z_base     = terrain_z - depth_below
    h_total    = z_top - z_base
    r_base     = r_top + h_total * np.tan(alpha_rad)
    h_presek   = z_top - terrain_z
    r_presek   = r_top + h_presek * np.tan(alpha_rad)

    def frustum_vol(r1, r2, h):
        return (np.pi * h / 3.0) * (r1**2 + r1*r2 + r2**2)

    v_total  = frustum_vol(r_top, r_base, h_total)
    v_iznad  = frustum_vol(r_top, r_presek, h_presek)
    v_ispod  = frustum_vol(r_presek, r_base, depth_below)
    a_gornji = np.pi * r_top**2
    a_presek = np.pi * r_presek**2
    a_baza   = np.pi * r_base**2

    assert abs(v_total - (v_iznad + v_ispod)) < 1.0

    N = 120
    theta = np.linspace(0, 2*np.pi, N)

    top_x   = cx + r_top    * np.cos(theta)
    top_y   = cy + r_top    * np.sin(theta)
    pres_x  = cx + r_presek * np.cos(theta)
    pres_y  = cy + r_presek * np.sin(theta)
    base_x  = cx + r_base   * np.cos(theta)
    base_y  = cy + r_base   * np.sin(theta)

    # Teren — ravan kvadrat
    margin = r_base * 2.0
    t_x = np.array([cx-margin, cx+margin, cx+margin, cx-margin, cx-margin])
    t_y = np.array([cy-margin, cy-margin, cy+margin, cy+margin, cy-margin])
    t_z = np.full(5, terrain_z)

    fig = go.Figure()

    # Teren (svetlo-plavi popunjen kvadrat)
    fig.add_trace(go.Scatter3d(
        x=t_x, y=t_y, z=t_z,
        mode="lines",
        line=dict(color="#4a8aaa", width=1),
        surfaceaxis=2,
        surfacecolor="rgba(122, 184, 212, 0.5)",
        name="Ravni teren",
        showlegend=True,
    ))

    # Mreza terena
    for i in np.linspace(cx-margin, cx+margin, 8):
        fig.add_trace(go.Scatter3d(
            x=[i, i], y=[cy-margin, cy+margin], z=[terrain_z, terrain_z],
            mode="lines", line=dict(color="#4a8aaa", width=0.5),
            showlegend=False, hoverinfo="skip",
        ))
    for j in np.linspace(cy-margin, cy+margin, 8):
        fig.add_trace(go.Scatter3d(
            x=[cx-margin, cx+margin], y=[j, j], z=[terrain_z, terrain_z],
            mode="lines", line=dict(color="#4a8aaa", width=0.5),
            showlegend=False, hoverinfo="skip",
        ))

    # Bocne strane frustuma IZNAD terena (zelene)
    x_boc, y_boc, z_boc = [], [], []
    for i in range(N-1):
        x_boc += [top_x[i],   pres_x[i],   pres_x[i+1], top_x[i+1],   None]
        y_boc += [top_y[i],   pres_y[i],   pres_y[i+1], top_y[i+1],   None]
        z_boc += [z_top,      terrain_z,   terrain_z,   z_top,         None]

    fig.add_trace(go.Scatter3d(
        x=x_boc, y=y_boc, z=z_boc,
        mode="lines",
        line=dict(color="rgba(34,170,68,0.4)", width=1),
        surfaceaxis=None,
        name="Frustum (iznad terena)",
        showlegend=True,
    ))

    # Bocne strane ISPOD terena (zelene, isprekidane)
    x_isp, y_isp, z_isp = [], [], []
    for i in range(N-1):
        x_isp += [pres_x[i],   base_x[i],   base_x[i+1], pres_x[i+1],   None]
        y_isp += [pres_y[i],   base_y[i],   base_y[i+1], pres_y[i+1],   None]
        z_isp += [terrain_z,   z_base,      z_base,      terrain_z,      None]

    fig.add_trace(go.Scatter3d(
        x=x_isp, y=y_isp, z=z_isp,
        mode="lines",
        line=dict(color="rgba(34,170,68,0.25)", width=1, dash="dot"),
        name="Frustum (ispod terena)",
        showlegend=True,
    ))

    # Gornji krug (tamno zelena linija)
    fig.add_trace(go.Scatter3d(
        x=np.append(top_x, top_x[0]),
        y=np.append(top_y, top_y[0]),
        z=np.full(N+1, z_top),
        mode="lines",
        line=dict(color="#116622", width=4),
        name=f"Gornji krug (R={r_top:.1f}m)",
    ))

    # Kontura preseka (crvena linija + popunjen disk)
    fig.add_trace(go.Scatter3d(
        x=np.append(pres_x, pres_x[0]),
        y=np.append(pres_y, pres_y[0]),
        z=np.full(N+1, terrain_z + 0.1),
        mode="lines",
        line=dict(color="#dd2222", width=5),
        name=f"Presek sa terenom (R_p={r_presek:.1f}m)",
    ))

    # Popunjen presek (fan trokuta kao Mesh3d)
    px_fan  = np.concatenate([[cx], pres_x])
    py_fan  = np.concatenate([[cy], pres_y])
    pz_fan  = np.full(N+1, terrain_z + 0.05)
    i_fan   = [0] * (N-1)
    j_fan   = list(range(1, N))
    k_fan   = list(range(2, N+1))
    fig.add_trace(go.Mesh3d(
        x=px_fan, y=py_fan, z=pz_fan,
        i=i_fan, j=j_fan, k=k_fan,
        color="#dd2222", opacity=0.75,
        name="Presek (popunjen)",
        showlegend=False,
    ))

    # Frustum bocna površina kao Mesh3d (iznad terena)
    x_m = np.concatenate([top_x, pres_x])
    y_m = np.concatenate([top_y, pres_y])
    z_m = np.concatenate([np.full(N, z_top), np.full(N, terrain_z)])
    im, jm, km = [], [], []
    for i in range(N-1):
        im += [i,     i+1,   i+N]
        jm += [i+1,   i+N+1, i+N+1]
        km += [i+N,   i+N,   i+1]
    fig.add_trace(go.Mesh3d(
        x=x_m, y=y_m, z=z_m,
        i=im, j=jm, k=km,
        color="#22aa44", opacity=0.40,
        name="Frustum površina",
        showlegend=True,
    ))

    # Gornji disk (Mesh3d)
    tx_fan = np.concatenate([[cx], top_x])
    ty_fan = np.concatenate([[cy], top_y])
    tz_fan = np.full(N+1, z_top)
    fig.add_trace(go.Mesh3d(
        x=tx_fan, y=ty_fan, z=tz_fan,
        i=i_fan, j=j_fan, k=k_fan,
        color="#22aa44", opacity=0.50,
        name="Gornji disk",
        showlegend=False,
    ))

    # Osa
    fig.add_trace(go.Scatter3d(
        x=[cx, cx], y=[cy, cy], z=[z_base, z_top],
        mode="lines",
        line=dict(color="white", width=1.5, dash="dash"),
        name="Osa",
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(
            text=f"Frustum  (R_top={r_top:.1f}m, α={alpha_deg:.1f}°)  ·  Ravni teren  ·  Presek",
            font=dict(size=14),
        ),
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z — visina (m)",
            aspectmode="data",
            bgcolor="rgba(10,12,20,1)",
            xaxis=dict(gridcolor="#223", showbackground=True,
                       backgroundcolor="rgba(10,12,20,1)"),
            yaxis=dict(gridcolor="#223", showbackground=True,
                       backgroundcolor="rgba(10,12,20,1)"),
            zaxis=dict(gridcolor="#223", showbackground=True,
                       backgroundcolor="rgba(10,12,20,1)"),
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
        ),
        height=640,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="rgba(20,20,30,0.85)", font=dict(size=11)),
    )

    proracun = dict(
        r_top=r_top, r_presek=r_presek, r_base=r_base,
        a_gornji=a_gornji, a_presek=a_presek, a_baza=a_baza,
        v_total=v_total, v_iznad=v_iznad, v_ispod=v_ispod,
        h_total=h_total, h_iznad=h_presek, h_ispod=depth_below,
    )
    return fig, proracun


def _frustum_plotly_neravni(
    cx: float, cy: float, z_top: float,
    r_top: float, alpha_deg: float,
    depth_below: float = 10.0,
    nx: int = 100, ny: int = 40,
    amp: float = 6.0,
) -> tuple[go.Figure, dict]:
    """Plotly 3D prikaz frustuma na NERAVNOM terenu (sinusoidalna aproksimacija)."""

    alpha_rad = np.radians(alpha_deg)

    # Teren
    margin = (r_top + 200) * 2.5
    x_lin  = np.linspace(cx - margin/2, cx + margin/2, nx)
    y_lin  = np.linspace(cy - margin/4, cy + margin/4, ny)
    X_t, Y_t = np.meshgrid(x_lin, y_lin)
    xl = X_t - cx; yl = Y_t - cy
    Z_t = (amp     * np.sin(0.008 * xl) * np.cos(0.012 * yl)
         + amp*0.5 * np.sin(0.018 * xl + 1.1)
         + amp*0.35* np.cos(0.009 * yl + 0.7))

    # Geometrija frustuma
    r_max_est  = r_top + (z_top - Z_t.min()) * np.tan(alpha_rad)
    dist_c     = np.sqrt((X_t - cx)**2 + (Y_t - cy)**2)
    zone_mask  = dist_c <= r_max_est * 1.1
    z_min_zona = Z_t[zone_mask].min() if zone_mask.any() else Z_t.min()
    z_base     = z_min_zona - depth_below
    h_total    = z_top - z_base
    r_base     = r_top + h_total * np.tan(alpha_rad)

    # Presek: tacke grida unutar frustuma na visini terena
    r_frust_at_terrain = r_top + (z_top - Z_t) * np.tan(alpha_rad)
    r_frust_at_terrain = np.where(Z_t <= z_top, r_frust_at_terrain, 0.0)
    inside = (dist_c <= r_frust_at_terrain) & (Z_t <= z_top)

    # Zapremina (numericki)
    dx = (x_lin[-1] - x_lin[0]) / (nx - 1)
    dy = (y_lin[-1] - y_lin[0]) / (ny - 1)
    z_frust_surf = np.where(
        inside,
        z_top - np.maximum(dist_c - r_top, 0.0) / np.tan(alpha_rad),
        0.0,
    )
    z_frust_surf = np.clip(z_frust_surf, z_base, z_top)
    dz_col = np.maximum(np.where(inside, z_frust_surf - Z_t, 0.0), 0.0)
    v_presek_num = np.sum(dz_col) * dx * dy

    # Zapremina celog frustuma
    def frustum_vol(r1, r2, h):
        return (np.pi * h / 3.0) * (r1**2 + r1*r2 + r2**2)
    v_total = frustum_vol(r_top, r_base, h_total)

    # Povrsina preseka
    dZdx, dZdy = np.gradient(Z_t, dx, dy)
    surf_elem  = np.sqrt(1.0 + dZdx**2 + dZdy**2)
    a_presek_2d = inside.sum() * dx * dy
    a_presek_3d = np.sum(surf_elem[inside]) * dx * dy

    N = 100
    theta  = np.linspace(0, 2*np.pi, N)
    top_x  = cx + r_top  * np.cos(theta);  top_y  = cy + r_top  * np.sin(theta)
    base_x = cx + r_base * np.cos(theta);  base_y = cy + r_base * np.sin(theta)

    fig = go.Figure()

    # Teren — Mesh3d bojano po visini
    from scipy.spatial import Delaunay as _Del
    v_flat = np.column_stack([X_t.ravel(), Y_t.ravel()])
    z_flat = Z_t.ravel()
    try:
        tri = _Del(v_flat)
        i_t, j_t, k_t = tri.simplices[:,0], tri.simplices[:,1], tri.simplices[:,2]
        intensity = (z_flat - z_flat.min()) / max(z_flat.max() - z_flat.min(), 1)
        fig.add_trace(go.Mesh3d(
            x=v_flat[:,0], y=v_flat[:,1], z=z_flat,
            i=i_t, j=j_t, k=k_t,
            intensity=intensity,
            colorscale=[
                [0.0,  "rgba(180,220,240,0.9)"],
                [0.4,  "rgba(100,170,210,0.9)"],
                [0.7,  "rgba(60,130,180,0.9)"],
                [1.0,  "rgba(30,80,140,0.9)"],
            ],
            showscale=True,
            colorbar=dict(title="Visina terena (m)", len=0.5),
            opacity=0.80,
            name="Neravni teren",
            hovertemplate="X: %{x:.0f}<br>Y: %{y:.0f}<br>Z: %{z:.2f}m<extra>Teren</extra>",
        ))
    except Exception:
        pass

    # Presek — crvene tacke + kontura
    IX = X_t[inside]; IY = Y_t[inside]; IZ = Z_t[inside]
    if len(IX) > 0:
        fig.add_trace(go.Scatter3d(
            x=IX, y=IY, z=IZ + 0.15,
            mode="markers",
            marker=dict(size=3, color="#dd2222", opacity=0.75),
            name=f"Presek zona (A_2D={a_presek_2d:,.0f}m²)",
        ))

    # Granica preseka
    from matplotlib.path import Path as MplPath
    boundary_mask = np.zeros_like(inside, dtype=bool)
    boundary_mask[1:-1, 1:-1] = (
        inside[1:-1,1:-1] & (
            ~inside[:-2,1:-1] | ~inside[2:,1:-1] |
            ~inside[1:-1,:-2] | ~inside[1:-1,2:]
        )
    )
    BX = X_t[boundary_mask]; BY = Y_t[boundary_mask]; BZ = Z_t[boundary_mask]
    if len(BX) > 0:
        ang = np.arctan2(BY - cy, BX - cx)
        si  = np.argsort(ang)
        BX, BY, BZ = BX[si], BY[si], BZ[si]
        fig.add_trace(go.Scatter3d(
            x=np.append(BX, BX[0]),
            y=np.append(BY, BY[0]),
            z=np.append(BZ, BZ[0]) + 0.2,
            mode="lines",
            line=dict(color="#ff2222", width=4),
            name="Granica preseka",
        ))

    # Frustum bocna površina (iznad terena — Mesh3d)
    x_m = np.concatenate([top_x, base_x])
    y_m = np.concatenate([top_y, base_y])
    z_m = np.concatenate([np.full(N, z_top), np.full(N, z_base)])
    im, jm, km = [], [], []
    for i in range(N-1):
        im += [i,   i+1, i+N];   jm += [i+1, i+N+1, i+N+1];   km += [i+N, i+N, i+1]
    fig.add_trace(go.Mesh3d(
        x=x_m, y=y_m, z=z_m,
        i=im, j=jm, k=km,
        color="#22aa44", opacity=0.38,
        name="Frustum",
        showlegend=True,
    ))

    # Gornji krug
    fig.add_trace(go.Scatter3d(
        x=np.append(top_x, top_x[0]),
        y=np.append(top_y, top_y[0]),
        z=np.full(N+1, z_top),
        mode="lines",
        line=dict(color="#116622", width=4),
        name=f"Gornji krug (R_top={r_top:.1f}m)",
    ))

    # Gornji disk
    tx_f = np.concatenate([[cx], top_x])
    ty_f = np.concatenate([[cy], top_y])
    tz_f = np.full(N+1, z_top)
    i_f  = [0]*(N-1); j_f = list(range(1,N)); k_f = list(range(2,N+1))
    fig.add_trace(go.Mesh3d(
        x=tx_f, y=ty_f, z=tz_f,
        i=i_f, j=j_f, k=k_f,
        color="#22aa44", opacity=0.55,
        name="Gornji disk",
        showlegend=False,
    ))

    # Osa
    fig.add_trace(go.Scatter3d(
        x=[cx, cx], y=[cy, cy], z=[z_base, z_top],
        mode="lines",
        line=dict(color="white", width=1.5, dash="dash"),
        name="Osa", showlegend=False,
    ))

    fig.update_layout(
        title=dict(
            text=f"Frustum  (R_top={r_top:.1f}m, α={alpha_deg:.1f}°)  ·  Neravni teren  ·  Presek",
            font=dict(size=14),
        ),
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
            bgcolor="rgba(10,12,20,1)",
            xaxis=dict(gridcolor="#223", showbackground=True,
                       backgroundcolor="rgba(10,12,20,1)"),
            yaxis=dict(gridcolor="#223", showbackground=True,
                       backgroundcolor="rgba(10,12,20,1)"),
            zaxis=dict(gridcolor="#223", showbackground=True,
                       backgroundcolor="rgba(10,12,20,1)"),
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
        ),
        height=640,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="rgba(20,20,30,0.85)", font=dict(size=11)),
    )

    proracun = dict(
        r_top=r_top, r_base=r_base,
        a_presek_2d=a_presek_2d, a_presek_3d=a_presek_3d,
        v_total=v_total, v_presek_num=v_presek_num,
        h_total=h_total, z_base=z_base, z_min_zona=z_min_zona,
    )
    return fig, proracun


with tab_frustum:
    st.subheader("Frustum — direktni prikaz")
    st.caption(
        "Generiše i prikazuje frustum (zarubljenu kupu) na zadanim parametrima. "
        "Nema Monte Carlo ni GA — samo geometrija, presek i proračun."
    )

    # --- Tip terena ---
    tip_terena = st.radio(
        "Tip terena",
        ["🟦 Ravni teren", "🌄 Neravni teren (sinusoidalni)"],
        horizontal=True,
    )

    st.divider()

    # --- Parametri frustuma ---
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Pozicija centra**")
        if st.session_state.podaci is not None:
            _pd = st.session_state.podaci
            _cx_def = float(_pd.centar_masa[0]) if _pd.centar_masa is not None else 4_970_000.0
            _cy_def = float(_pd.centar_masa[1]) if _pd.centar_masa is not None else 6_388_500.0
            _cz_def = float(_pd.teren.vertices[:, 2].mean()) + 50.0
        else:
            _cx_def, _cy_def, _cz_def = 4_970_000.0, 6_388_500.0, 75.0

        fr_cx    = st.number_input("Centar X", value=_cx_def, format="%.2f", key="fr_cx")
        fr_cy    = st.number_input("Centar Y", value=_cy_def, format="%.2f", key="fr_cy")
        fr_z_top = st.number_input("Visina gornjeg kruga Z_top (m)",
                                   value=_cz_def, format="%.2f", key="fr_ztop")

    with col_r:
        st.markdown("**Dimenzije**")
        fr_r_top   = st.number_input("Poluprecnik gornjeg kruga R_top (m)",
                                     min_value=1.0, value=15.0, step=1.0, key="fr_rtop")
        fr_alpha   = st.number_input("Poluugao strana α (°)",
                                     min_value=1.0, max_value=80.0,
                                     value=25.0, step=0.5, key="fr_alpha")
        fr_depth   = st.number_input("Dubina baze ispod min. terena (m)",
                                     min_value=1.0, value=10.0, step=1.0, key="fr_depth")
        if tip_terena == "🟦 Ravni teren":
            fr_tz = st.number_input("Nivo terena Z (m)",
                                    value=0.0, format="%.2f", key="fr_tz")
        else:
            fr_amp = st.slider("Amplituda neravnina terena (m)",
                               min_value=0.5, max_value=20.0, value=6.0, step=0.5,
                               key="fr_amp")

    st.divider()
    btn_frustum = st.button("▶ Generiši frustum", type="primary",
                            use_container_width=True, key="btn_frustum")

    if btn_frustum:
        with st.spinner("Računam geometriju i generišem prikaz..."):
            try:
                if tip_terena == "🟦 Ravni teren":
                    fig_fr, prac = _frustum_plotly_ravan(
                        cx=fr_cx, cy=fr_cy, z_top=fr_z_top,
                        r_top=fr_r_top, alpha_deg=fr_alpha,
                        terrain_z=fr_tz, depth_below=fr_depth,
                    )
                    # Metrike
                    st.success("Frustum generisan!")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("R_top (gornji)",   f"{prac['r_top']:.2f} m")
                    m2.metric("R_presek (teren)", f"{prac['r_presek']:.2f} m")
                    m3.metric("R_base (dno)",     f"{prac['r_base']:.2f} m")
                    m4.metric("Visina H",          f"{prac['h_total']:.2f} m")

                    m5, m6, m7, m8 = st.columns(4)
                    m5.metric("A gornji krug",    f"{prac['a_gornji']:,.1f} m²")
                    m6.metric("A presek",          f"{prac['a_presek']:,.1f} m²")
                    m7.metric("A baza",            f"{prac['a_baza']:,.1f} m²")
                    m8.metric("H iznad terena",    f"{prac['h_iznad']:.1f} m")

                    st.divider()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Zapremina CELOG frustuma", f"{prac['v_total']:,.1f} m³")
                    c2.metric("Zapremina IZNAD terena",   f"{prac['v_iznad']:,.1f} m³  "
                              f"({prac['v_iznad']/prac['v_total']*100:.1f}%)")
                    c3.metric("Zapremina ISPOD terena",   f"{prac['v_ispod']:,.1f} m³  "
                              f"({prac['v_ispod']/prac['v_total']*100:.1f}%)")

                else:
                    fig_fr, prac = _frustum_plotly_neravni(
                        cx=fr_cx, cy=fr_cy, z_top=fr_z_top,
                        r_top=fr_r_top, alpha_deg=fr_alpha,
                        depth_below=fr_depth, amp=fr_amp,
                    )
                    st.success("Frustum na neravnom terenu generisan!")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("R_top (gornji)",    f"{prac['r_top']:.2f} m")
                    m2.metric("R_base (dno)",      f"{prac['r_base']:.2f} m")
                    m3.metric("Z baze",             f"{prac['z_base']:.2f} m")
                    m4.metric("Min. teren u zoni",  f"{prac['z_min_zona']:.2f} m")

                    st.divider()
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("A presek (2D proj.)",  f"{prac['a_presek_2d']:,.1f} m²")
                    c2.metric("A presek (3D nagib)",  f"{prac['a_presek_3d']:,.1f} m²")
                    c3.metric("V celog frustuma",     f"{prac['v_total']:,.1f} m³")
                    c4.metric("V presek → vrh",       f"{prac['v_presek_num']:,.1f} m³  "
                              f"({prac['v_presek_num']/prac['v_total']*100:.1f}%)")

                st.plotly_chart(fig_fr, use_container_width=True)

            except Exception as e:
                st.error(f"Greška: {e}")
                import traceback
                st.code(traceback.format_exc())
