import os
import math
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import pydeck as pdk


# =========================
# CONFIG
# =========================
WINDY_ENDPOINT = "https://api.windy.com/api/point-forecast/v2"
MODEL = "gfs"
PARAMETERS = ["wind", "windGust", "cape"]

# IMPORTANTE:
# Windy NO acepta 250hPa (≈FL340). Solo acepta: surface, 1000h, 950h, ..., 300h, 200h, 150h, etc.
# Por eso, aproximamos FL340 interpolando en log(p) entre 300hPa y 200hPa hacia 250hPa.
LEVELS = ["surface", "400h", "300h", "200h", "150h"]

AIRPORTS = {
    "MAD": {"lat": 40.4936, "lon": -3.5668, "tz": "Europe/Madrid"},
    "TFS": {"lat": 28.0445, "lon": -16.5725, "tz": "Atlantic/Canary"},
}

# Vuelos "típicos" (editables). Horas locales de origen/destino.
TYPICAL_FLIGHTS = {
    ("MAD", "TFS"): [
        ("07:10", "09:00"),
        ("12:20", "14:10"),
        ("19:10", "21:00"),
        ("22:45", "01:00"),
    ],
    ("TFS", "MAD"): [
        ("07:00", "10:45"),
        ("11:30", "15:15"),
        ("17:30", "21:15"),
        ("20:30", "00:15"),
    ],
}

DEFAULT_INTERVALS = [10, 15]

# Umbrales (ajústalos con tu experiencia / calibración posterior)
JET_ORANGE = 140
JET_RED = 180

SHEAR_ORANGE = 25
SHEAR_RED = 35

CAPE_ORANGE = 800
CAPE_RED = 1400

# Orografía Teide cerca de llegada a TFS
TFS_ORO_START_FRAC = 0.75
SURF_ORO_ORANGE = 35
SURF_ORO_RED = 45
GUST_ORO_ORANGE = 50
GUST_ORO_RED = 65


# =========================
# HELPERS GEO / MATH
# =========================
def rad(x): return x * math.pi / 180.0
def deg(x): return x * 180.0 / math.pi

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = rad(lat2 - lat1)
    dlon = rad(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(rad(lat1))*math.cos(rad(lat2))*math.sin(dlon/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R*c

def intermediate_point(lat1, lon1, lat2, lon2, frac):
    # Great-circle interpolation
    d = haversine_km(lat1, lon1, lat2, lon2) / 6371.0
    if d == 0:
        return lat1, lon1
    φ1, λ1 = rad(lat1), rad(lon1)
    φ2, λ2 = rad(lat2), rad(lon2)

    a = math.sin((1-frac)*d)/math.sin(d)
    b = math.sin(frac*d)/math.sin(d)

    x = a*math.cos(φ1)*math.cos(λ1) + b*math.cos(φ2)*math.cos(λ2)
    y = a*math.cos(φ1)*math.sin(λ1) + b*math.cos(φ2)*math.sin(λ2)
    z = a*math.sin(φ1) + b*math.sin(φ2)

    φi = math.atan2(z, math.sqrt(x*x + y*y))
    λi = math.atan2(y, x)
    return deg(φi), deg(λi)

def wind_speed_kmh(u, v):
    return math.sqrt(u*u + v*v) * 3.6

def interp_logp(p1, v1, p2, v2, p_target):
    """Interpolación lineal en log(p) (p en hPa)."""
    lp1, lp2, lpt = math.log(p1), math.log(p2), math.log(p_target)
    t = (lp1 - lpt) / (lp1 - lp2)
    return v1 + (v2 - v1) * t

def clamp(x, a, b):
    return max(a, min(b, x))


# =========================
# TIME HELPERS (TZ aware)
# =========================
def parse_local_dt(date_str, hhmm, tz_name):
    hh, mm = map(int, hhmm.split(":"))
    base = datetime.strptime(date_str, "%Y-%m-%d")
    return datetime(base.year, base.month, base.day, hh, mm, tzinfo=ZoneInfo(tz_name))

def dep_arr_utc(org, des, date_str, dep_hhmm, arr_hhmm):
    dep = parse_local_dt(date_str, dep_hhmm, AIRPORTS[org]["tz"]).astimezone(ZoneInfo("UTC"))
    arr = parse_local_dt(date_str, arr_hhmm, AIRPORTS[des]["tz"]).astimezone(ZoneInfo("UTC"))
    if arr <= dep:
        arr += timedelta(days=1)
    return dep, arr


# =========================
# OPTIONS
# =========================
def build_options(date_str):
    opts = []
    for (org, des), flights in TYPICAL_FLIGHTS.items():
        for dep, arr in flights:
            label = f"{date_str}  {org}→{des}  (dep {dep} local {org}, arr {arr} local {des})"
            opts.append((label, org, des, dep, arr))
    return opts


# =========================
# WINDY API (CACHE)
# =========================
@st.cache_data(ttl=60*10)  # 10 min cache
def fetch_windy(lat, lon, api_key):
    payload = {
        "lat": lat,
        "lon": lon,
        "model": MODEL,
        "parameters": PARAMETERS,
        "levels": LEVELS,
        "key": api_key
    }
    r = requests.post(WINDY_ENDPOINT, json=payload, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Windy {r.status_code}: {r.text[:200]}")
    return r.json()

def closest_idx(ts_ms, target_dt_utc):
    target_ms = int(target_dt_utc.timestamp() * 1000)
    return min(range(len(ts_ms)), key=lambda i: abs(ts_ms[i] - target_ms))


# =========================
# SCORING (CAT / shear / CAPE / orografía)
# =========================
def score_point(ws_fl340, ws200, shear, cape, surf_ws, gust, frac, des):
    intensity = 2
    color = "green"
    why = []

    jet = max(ws_fl340, ws200)

    if jet >= JET_RED:
        intensity = max(intensity, 8); color = "red"
        why.append(f"Jet muy fuerte (~{jet:.0f} km/h).")
    elif jet >= JET_ORANGE:
        intensity = max(intensity, 6); color = "orange"
        why.append(f"Jet fuerte (~{jet:.0f} km/h).")

    if shear >= SHEAR_RED:
        intensity = max(intensity, 8); color = "red"
        why.append(f"Cizalladura alta (~{shear:.1f} km/h/km) → CAT probable.")
    elif shear >= SHEAR_ORANGE:
        intensity = max(intensity, 6)
        if color != "red": color = "orange"
        why.append(f"Cizalladura moderada (~{shear:.1f} km/h/km).")

    if cape >= CAPE_RED:
        intensity = max(intensity, 8); color = "red"
        why.append(f"CAPE alto ({cape:.0f}) → convectiva.")
    elif cape >= CAPE_ORANGE:
        intensity = max(intensity, 6)
        if color != "red": color = "orange"
        why.append(f"CAPE moderado ({cape:.0f}).")

    # Orografía Teide (heurística) en el tramo final hacia TFS
    if des == "TFS" and frac >= TFS_ORO_START_FRAC:
        if surf_ws >= SURF_ORO_RED or gust >= GUST_ORO_RED:
            intensity = max(intensity, 8); color = "red"
            why.append(f"Orografía Teide: viento/rachas fuertes (surf {surf_ws:.0f}, gust {gust:.0f}).")
        elif surf_ws >= SURF_ORO_ORANGE or gust >= GUST_ORO_ORANGE:
            intensity = max(intensity, 6)
            if color != "red": color = "orange"
            why.append(f"Orografía Teide: viento/rachas moderadas (surf {surf_ws:.0f}, gust {gust:.0f}).")
        else:
            why.append("Zona llegada a TFS (posible orografía).")

    if not why:
        why.append("Condiciones estables.")

    prob = int(clamp(10 + intensity * 10, 15, 90))
    status = "Baja" if intensity <= 3 else ("Moderada" if intensity <= 7 else "Alta")
    return intensity, prob, color, status, " ".join(why)

def rgba_for_intensity(x):
    if pd.isna(x):
        return [220, 0, 0, 180]
    if x <= 3:
        return [0, 170, 80, 160]
    if x <= 7:
        return [255, 140, 0, 170]
    return [220, 0, 0, 180]


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Turbli++ MAD↔TFS", layout="wide")
st.title("🌪️ Turbli++ (MAD ↔ TFS) — Predicción profesional")
st.caption("Mapa + gráfico + leyenda. Intervalo 10/15 min. FL340 aprox (250h interpolado entre 300h y 200h).")

# Password opcional (recomendado en cloud)
APP_PASSWORD = os.getenv("APP_PASSWORD", "").strip()
if APP_PASSWORD:
    pw = st.text_input("Contraseña", type="password")
    if pw != APP_PASSWORD:
        st.stop()

api_key = os.getenv("WINDY_API_KEY", "").strip()
if not api_key:
    st.error("Falta WINDY_API_KEY en variables de entorno (Render/Fly).")
    st.stop()

colA, colB, colC, colD = st.columns([1, 1, 1, 1])
with colA:
    interval = st.selectbox("Intervalo (min)", DEFAULT_INTERVALS, index=1)
with colB:
    now = datetime.now(ZoneInfo("Europe/Madrid"))
    today = now.strftime("%Y-%m-%d")
    tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    date_str = st.selectbox("Fecha", [today, tomorrow], index=0)
with colC:
    mode = st.selectbox("Ruta", ["MAD→TFS", "TFS→MAD", "AMBOS"], index=2)
with colD:
    autoplay = st.checkbox("Auto refresco (15 min)", value=False)

options = build_options(date_str)
if mode == "MAD→TFS":
    options = [o for o in options if o[1] == "MAD" and o[2] == "TFS"]
elif mode == "TFS→MAD":
    options = [o for o in options if o[1] == "TFS" and o[2] == "MAD"]

label = st.selectbox("Vuelo (horarios típicos)", [o[0] for o in options], index=0)

selected = [o for o in options if o[0] == label]
if mode == "AMBOS":
    selected = options

run = st.button("Calcular ahora")
if autoplay:
    run = True

if run:
    all_dfs = []

    for (lab, org, des, dep_hhmm, arr_hhmm) in selected:
        dep_utc, arr_utc = dep_arr_utc(org, des, date_str, dep_hhmm, arr_hhmm)
        duration_min = int((arr_utc - dep_utc).total_seconds() / 60)
        n_points = max(duration_min // interval + 1, 2)

        org_lat, org_lon = AIRPORTS[org]["lat"], AIRPORTS[org]["lon"]
        des_lat, des_lon = AIRPORTS[des]["lat"], AIRPORTS[des]["lon"]

        st.markdown(f"### {org}→{des} · {date_str} · dep {dep_hhmm} (local {org}) · {n_points} puntos")
        prog = st.progress(0)
        rows = []

        for i in range(n_points):
            frac = i / (n_points - 1)
            lat, lon = intermediate_point(org_lat, org_lon, des_lat, des_lon, frac)
            t_utc = dep_utc + timedelta(minutes=i * interval)

            try:
                data = fetch_windy(lat, lon, api_key)
                ts = data.get("ts", [])
                idx = closest_idx(ts, t_utc)

                # vientos 300h / 200h / 400h
                u300 = data.get("wind_u-300h", [0])[idx]
                v300 = data.get("wind_v-300h", [0])[idx]
                u200 = data.get("wind_u-200h", [0])[idx]
                v200 = data.get("wind_v-200h", [0])[idx]
                u400 = data.get("wind_u-400h", [0])[idx]
                v400 = data.get("wind_v-400h", [0])[idx]

                ws300 = wind_speed_kmh(u300, v300)
                ws200 = wind_speed_kmh(u200, v200)
                ws400 = wind_speed_kmh(u400, v400)

                # FL340~ (250h) interpolado
                u250 = interp_logp(300, u300, 200, u200, 250)
                v250 = interp_logp(300, v300, 200, v200, 250)
                ws250 = wind_speed_kmh(u250, v250)

                # shear proxy
                shear = abs(ws300 - ws200) / 3.0

                # superficie
                us = data.get("wind_u-surface", [0])[idx]
                vs = data.get("wind_v-surface", [0])[idx]
                surf_ws = wind_speed_kmh(us, vs)
                gust = data.get("gust-surface", [0])[idx] * 3.6
                cape = data.get("cape-surface", [0])[idx]

                intensity, prob, color, status, why = score_point(
                    ws250, ws200, shear, cape, surf_ws, gust, frac, des
                )

                rows.append({
                    "route": f"{org}->{des}",
                    "min": i * interval,
                    "utc": t_utc.strftime("%H:%M"),
                    "lat": lat,
                    "lon": lon,
                    "ws_fl340": round(ws250, 1),
                    "ws200": round(ws200, 1),
                    "ws300": round(ws300, 1),
                    "ws400": round(ws400, 1),
                    "shear": round(shear, 2),
                    "cape": float(cape) if cape is not None else None,
                    "surf_ws": round(surf_ws, 1),
                    "gust": round(gust, 1),
                    "intensity": int(intensity),
                    "prob": int(prob),
                    "color": color,
                    "status": status,
                    "why": why
                })

            except Exception as e:
                rows.append({
                    "route": f"{org}->{des}",
                    "min": i * interval,
                    "utc": t_utc.strftime("%H:%M"),
                    "lat": lat,
                    "lon": lon,
                    "ws_fl340": None,
                    "ws200": None,
                    "ws300": None,
                    "ws400": None,
                    "shear": None,
                    "cape": None,
                    "surf_ws": None,
                    "gust": None,
                    "intensity": None,
                    "prob": None,
                    "color": "red",
                    "status": "Error",
                    "why": str(e)
                })

            prog.progress((i + 1) / n_points)
            time.sleep(0.02)

        df = pd.DataFrame(rows)
        all_dfs.append(df)

    result = pd.concat(all_dfs, ignore_index=True)
    result["rgba"] = result["intensity"].apply(rgba_for_intensity)

    # ===== MAPA =====
    st.subheader("🗺️ Mapa (ruta + puntos cada intervalo)")
    view_state = pdk.ViewState(
        latitude=float(result["lat"].mean()),
        longitude=float(result["lon"].mean()),
        zoom=4.6
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=result,
        get_position=["lon", "lat"],
        get_radius=18000,
        get_fill_color="rgba",
        pickable=True
    )

    tooltip = {
        "html": "<b>{route}</b> · {utc}Z<br/>"
                "Intensidad: <b>{intensity}</b>/10 ({prob}%)<br/>"
                "Viento FL340~: {ws_fl340} km/h<br/>"
                "Shear: {shear} km/h/km · CAPE: {cape}<br/>"
                "<i>{why}</i>",
        "style": {"backgroundColor": "rgba(20,20,20,0.85)", "color": "white"}
    }

    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))

    # ===== GRÁFICO =====
    st.subheader("📈 Intensidad (0–10) por tiempo")
    fig = px.line(
        result, x="min", y="intensity", color="route", markers=True,
        hover_data=["utc", "ws_fl340", "shear", "cape", "status"]
    )
    fig.update_yaxes(range=[0, 10])
    st.plotly_chart(fig, use_container_width=True)

    # ===== RESUMEN =====
    st.subheader("🧠 Resumen")
    max_int = int(result["intensity"].max(skipna=True) or 0)
    max_prob = int(result["prob"].max(skipna=True) or 0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Máx. intensidad", max_int)
    c2.metric("Máx. probabilidad", f"{max_prob}%")
    c3.metric("Puntos calculados", len(result))

    st.write("**Interpretación rápida**")
    st.write("- **0–3**: Baja · **4–7**: Moderada · **8–10**: Alta")
    st.write("- Causas: **Jet** (viento en altura), **Cizalladura** (CAT), **CAPE** (convectiva), **Orografía** en llegada a TFS")

    # ===== TABLA =====
    st.subheader("📋 Datos")
    st.dataframe(result, use_container_width=True)

    # Auto refresh
    if autoplay:
        st.info("Auto refresco activo: recargando en 15 min…")
        time.sleep(15 * 60)
        st.experimental_rerun()