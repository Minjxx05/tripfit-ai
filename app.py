import io
import json
import time
import urllib.parse
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from PIL import Image

from google import genai
from google.genai import errors as genai_errors


# ======================
# Config
# ======================
TEXT_MODEL = "gemini-2.5-flash"
IMAGE_MODEL = "gemini-2.5-flash-image"

UNSPLASH_SOURCE = "https://source.unsplash.com/1600x1200/?"  # key 없이 랜덤 이미지


# ======================
# Style (깔끔 + 잘림 방지)
# ======================
st.set_page_config(page_title="Tripfit", layout="wide")
st.markdown(
    """
<style>
.block-container { padding-top: .5rem; max-width: none !important; }
header, footer { visibility: hidden; height: 0; }
h1, h2, h3 { line-height: 1.05; margin: .2rem 0; }
.title { font-size: clamp(22px, 2.6vw, 34px); font-weight: 900; letter-spacing: -0.02em; }
.pillbar { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin: 8px 0 10px; }
.pill {
  padding: 8px 10px; border-radius: 999px;
  background: rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.06);
  font-size: 0.9rem;
}
.card {
  background: rgba(255,255,255,0.82);
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 18px;
  padding: 12px 12px;
  box-shadow: 0 10px 34px rgba(0,0,0,0.04);
}
.wxgrid { display:grid; grid-template-columns: repeat(5, minmax(130px, 1fr)); gap:10px; }
.wx {
  background: rgba(255,255,255,0.9);
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 16px;
  padding: 10px;
}
.wx .d { font-weight: 800; font-size: .92rem; }
.wx .t { font-weight: 900; font-size: 1.05rem; margin-top: 4px; }
.wx .s { opacity: .65; font-size: .82rem; margin-top: 4px; }
.small { opacity: .55; font-size: .85rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ======================
# Helpers
# ======================
def get_api_key() -> Optional[str]:
    k = st.session_state.get("api_key_input")
    if k and k.strip():
        return k.strip()
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return None


def gemini_client() -> genai.Client:
    k = get_api_key()
    if not k:
        raise RuntimeError("NO_KEY")
    return genai.Client(api_key=k)


def geocode_city(city: str) -> Optional[Tuple[float, float, str, str]]:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    r = requests.get(
        url,
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    results = data.get("results") or []
    if not results:
        return None
    it = results[0]
    return float(it["latitude"]), float(it["longitude"]), it.get("name", city), it.get("country", "")


def forecast_daily(lat: float, lon: float, start: date, end: date) -> Dict[str, Any]:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,windspeed_10m_max",
        "timezone": "auto",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def _rain_icon(pop: float) -> str:
    if pop >= 70:
        return "🌧️"
    if pop >= 40:
        return "🌦️"
    if pop >= 20:
        return "☁️"
    return "☀️"


def weather_cards(f: Dict[str, Any]) -> List[Dict[str, Any]]:
    d = f.get("daily", {}) or {}
    times = d.get("time", []) or []
    tmax = d.get("temperature_2m_max", []) or []
    tmin = d.get("temperature_2m_min", []) or []
    pop = d.get("precipitation_probability_max", []) or []
    wind = d.get("windspeed_10m_max", []) or []

    out = []
    for i in range(len(times)):
        out.append(
            {
                "date": times[i],
                "icon": _rain_icon(float(pop[i])),
                "tmin": int(round(float(tmin[i]))),
                "tmax": int(round(float(tmax[i]))),
                "pop": int(round(float(pop[i]))),
                "wind": int(round(float(wind[i]))),
            }
        )
    return out


@st.cache_data(show_spinner=False, ttl=60 * 10)
def cached_weather(city: str, start_iso: str, end_iso: str) -> Dict[str, Any]:
    geo = geocode_city(city)
    if not geo:
        return {"ok": False}
    lat, lon, name, country = geo
    f = forecast_daily(lat, lon, date.fromisoformat(start_iso), date.fromisoformat(end_iso))
    return {"ok": True, "place": f"{name}, {country}", "cards": weather_cards(f)}


def build_mood_prompts(city: str, season: str, vibe: str) -> List[str]:
    # 4-shot 고정 (일관된 무드)
    return [
        f"Photorealistic travel street scene in {city} during {season}. Vibe: {vibe}. cinematic light, no text, high detail.",
        f"Photorealistic travel outfit flat-lay. Destination: {city}. Season: {season}. Vibe: {vibe}. 7-9 items, no text, high detail.",
        f"Photorealistic candid travel moment in {city}. Season: {season}. Vibe: {vibe}. lifestyle, no text, high detail.",
        f"Photorealistic travel editorial inspired by {city}. Season: {season}. Vibe: {vibe}. premium look, no text, high detail.",
    ]


def gemini_images(prompts: List[str]) -> List[Image.Image]:
    client = gemini_client()
    imgs: List[Image.Image] = []
    for p in prompts:
        resp = client.models.generate_content(model=IMAGE_MODEL, contents=[p])
        parts = getattr(resp, "parts", None)
        if not parts and hasattr(resp, "candidates") and resp.candidates:
            parts = resp.candidates[0].content.parts

        got = False
        for part in parts or []:
            if getattr(part, "inline_data", None) is not None:
                imgs.append(part.as_image())
                got = True
                break
        if not got:
            # 한 장 실패해도 계속
            continue
    return imgs


def fetch_unsplash_images(city: str, vibe: str, season: str) -> List[Image.Image]:
    # 키 없이도 무조건 이미지가 옴 (랜덤). 4장 확보용.
    queries = [
        f"{city},travel,street,{vibe}",
        f"{city},architecture,city,{season}",
        f"{city},cafe,local,{vibe}",
        f"{city},night,cinematic,{vibe}",
    ]
    imgs: List[Image.Image] = []
    for i, q in enumerate(queries):
        # Unsplash source는 동일 쿼리면 캐시가 걸릴 수 있어 sig로 분리
        url = UNSPLASH_SOURCE + urllib.parse.quote(q) + f"&sig={int(time.time())}_{i}"
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        imgs.append(Image.open(io.BytesIO(r.content)).convert("RGB"))
    return imgs


def itinerary_json(city: str, start: date, end: date, vibe: str, prefs: List[str], wx: List[Dict[str, Any]]) -> Dict[str, Any]:
    # 텍스트는 대부분 키에서 잘 됨
    client = gemini_client()

    weather_lines = []
    for w in wx[:10]:
        weather_lines.append(f"{w['date']}: {w['tmin']}~{w['tmax']}C rain{w['pop']}% wind{w['wind']}kmh")
    weather_text = "\n".join(weather_lines) if weather_lines else "N/A"

    prefs_text = ", ".join(prefs) if prefs else "any"

    prompt = f"""
Return ONLY valid JSON.

Make a day-by-day itinerary.

Inputs:
- city: {city}
- dates: {start.isoformat()} to {end.isoformat()}
- vibe: {vibe}
- preferences: {prefs_text}
- weather:
{weather_text}

Schema:
{{
  "days": [
    {{
      "date": "YYYY-MM-DD",
      "theme": "short",
      "blocks": [
        {{"time":"09:00","title":"...","note":"..."}},
        {{"time":"12:30","title":"...","note":"..."}},
        {{"time":"16:00","title":"...","note":"..."}},
        {{"time":"20:00","title":"...","note":"..."}}
      ],
      "weather_tip": "one line"
    }}
  ]
}}
"""
    resp = client.models.generate_content(model=TEXT_MODEL, contents=[prompt])
    txt = getattr(resp, "text", "") or ""
    return json.loads(txt[txt.find("{") : txt.rfind("}") + 1])


# ======================
# App state
# ======================
st.session_state.setdefault("mood_imgs", [])
st.session_state.setdefault("mood_source", "")  # "gemini" or "unsplash"
st.session_state.setdefault("wx_place", "")
st.session_state.setdefault("wx_cards", [])
st.session_state.setdefault("plan", None)


# ======================
# Sidebar (조용하게)
# ======================
with st.sidebar:
    st.text_input("Gemini API Key", type="password", key="api_key_input", label_visibility="collapsed", placeholder="Gemini API Key")
    st.markdown("---")
    city = st.text_input("City", value="Tokyo")
    c1, c2 = st.columns(2)
    start_d = c1.date_input("From", value=date.today() + timedelta(days=7))
    end_d = c2.date_input("To", value=date.today() + timedelta(days=10))
    vibe = st.text_input("Vibe", value="clean, cinematic, warm tones")
    season = st.text_input("Season", value="current season")
    prefs = st.multiselect("Preferences", ["food/cafe", "museum", "shopping", "nature", "night view", "local", "relax"], default=["food/cafe", "museum"])


# ======================
# Top Bar
# ======================
st.markdown('<div class="title">Tripfit ✈️</div>', unsafe_allow_html=True)

# 자동 날씨 (버튼 없이)
wx = cached_weather(city, start_d.isoformat(), end_d.isoformat())
if wx.get("ok"):
    st.session_state.wx_place = wx["place"]
    st.session_state.wx_cards = wx["cards"]
else:
    st.session_state.wx_place = ""
    st.session_state.wx_cards = []


# ======================
# Compact Weather row (텍스트 최소)
# ======================
if st.session_state.wx_cards:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"<div class='small'>{st.session_state.wx_place}</div>", unsafe_allow_html=True)
    # 최대 5일만
    cards = st.session_state.wx_cards[:5]
    st.markdown('<div class="wxgrid">', unsafe_allow_html=True)
    for w in cards:
        st.markdown(
            f"""
<div class="wx">
  <div class="d">{w["date"]} {w["icon"]}</div>
  <div class="t">{w["tmin"]}°~{w["tmax"]}°</div>
  <div class="s">☔ {w["pop"]}% · 💨 {w["wind"]}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ======================
# Buttons (텍스트 없이)
# ======================
b1, b2 = st.columns([1, 1])
with b1:
    open_mood = st.button("Moodboard", type="primary", use_container_width=True)
with b2:
    make_plan = st.button("Itinerary", use_container_width=True)


# ======================
# Moodboard Popup (무조건 이미지)
# ======================
@st.dialog("Moodboard", width="large")
def moodboard_dialog():
    # 이미 생성해둔 게 있으면 바로 보여주기
    if not st.session_state.mood_imgs:
        # 생성
        prompts = build_mood_prompts(city, season, vibe)

        # 1) Gemini 시도
        try:
            if not get_api_key():
                raise RuntimeError("NO_KEY")
            imgs = gemini_images(prompts)
            if len(imgs) >= 4:
                st.session_state.mood_imgs = imgs[:4]
                st.session_state.mood_source = "gemini"
            else:
                # 부족하면 fallback
                raise RuntimeError("EMPTY_OR_PARTIAL")
        except Exception:
            # 2) 무조건 되는 fallback (Unsplash)
            try:
                imgs = fetch_unsplash_images(city, vibe, season)
                st.session_state.mood_imgs = imgs[:4]
                st.session_state.mood_source = "unsplash"
            except Exception:
                st.session_state.mood_imgs = []
                st.session_state.mood_source = ""

    imgs = st.session_state.mood_imgs
    if imgs:
        c1, c2 = st.columns(2)
        c1.image(imgs[0], use_container_width=True)
        c2.image(imgs[1], use_container_width=True)
        c1.image(imgs[2], use_container_width=True)
        c2.image(imgs[3], use_container_width=True)

        # 화면에 문장 띄우지 말랬으니, 배지는 아주 작게만
        if st.session_state.mood_source == "unsplash":
            st.caption("source: photo")
    else:
        # 여기까지 오면 네트워크 문제 급이라 짧게만
        st.error("failed")


if open_mood:
    moodboard_dialog()


# ======================
# Itinerary (화면 텍스트 최소 + 결과만)
# ======================
if make_plan:
    try:
        if not get_api_key():
            st.toast("need key", icon="🔑")
        else:
            wx_cards_state = st.session_state.wx_cards
            st.session_state.plan = itinerary_json(city, start_d, end_d, vibe, prefs, wx_cards_state)
            st.toast("done", icon="✅")
    except genai_errors.ClientError:
        st.toast("text api blocked", icon="⚠️")
    except Exception:
        st.toast("failed", icon="⚠️")

plan = st.session_state.plan
if plan and plan.get("days"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    for day in plan["days"]:
        # 날짜/테마만 굵게, 나머지 최소
        st.markdown(f"**{day.get('date','')} — {day.get('theme','')}**")
        for b in day.get("blocks", []):
            st.write(f"- {b.get('time','')} {b.get('title','')} · {b.get('note','')}")
        tip = day.get("weather_tip")
        if tip:
            st.caption(tip)
        st.divider()
    st.markdown("</div>", unsafe_allow_html=True)
