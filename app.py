import json
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from PIL import Image

from google import genai
from google.genai import errors as genai_errors


# =====================
# CONFIG
# =====================
TEXT_MODEL_DEFAULT = "gemini-2.5-flash"
IMAGE_MODEL_DEFAULT = "gemini-2.5-flash-image"  # Nano Banana 계열


# =====================
# JSON helper
# =====================
def _safe_json_loads(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 3:
            t = parts[1]
    t = t.strip()
    i, j = t.find("{"), t.rfind("}")
    if i != -1 and j != -1 and j > i:
        t = t[i : j + 1]
    return json.loads(t)


# =====================
# Key / Client
# =====================
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
        raise RuntimeError("Gemini API Key 없음")
    return genai.Client(api_key=k)


def call_text(prompt: str, model: str, temperature: float = 0.6) -> str:
    client = gemini_client()
    resp = client.models.generate_content(model=model, contents=[prompt])
    return getattr(resp, "text", "") or ""


def call_json(prompt: str, model: str, retries: int = 2) -> Dict[str, Any]:
    rule = "반드시 유효한 JSON만 출력. 다른 텍스트/설명/마크다운/코드펜스 금지."
    last = None
    for _ in range(retries + 1):
        try:
            txt = call_text(rule + "\n" + prompt, model=model)
            return _safe_json_loads(txt)
        except Exception as e:
            last = e
    raise RuntimeError(f"JSON 파싱 실패: {last}")


# =====================
# Weather (Open-Meteo)
# =====================
def geocode_city(city: str) -> Optional[Tuple[float, float, str, str]]:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    r = requests.get(url, params={"name": city, "count": 1, "language": "ko", "format": "json"}, timeout=20)
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
        out.append({
            "date": times[i],
            "icon": _rain_icon(float(pop[i])),
            "tmin": int(round(float(tmin[i]))),
            "tmax": int(round(float(tmax[i]))),
            "pop": int(round(float(pop[i]))),
            "wind": int(round(float(wind[i]))),
        })
    return out


# =====================
# Moodboard
# =====================
def build_prompts(city: str, season: str, style: str, vibe: str) -> List[str]:
    return [
        f"Photorealistic street-style fashion photo in {city} during {season}. Style: {style}. Vibe: {vibe}. Full body, natural light, no text, high detail.",
        f"Photorealistic outfit flat-lay on warm neutral background. Destination: {city}, season: {season}. Style: {style}. Include 7-9 items, no text, high detail.",
        f"Photorealistic candid travel moment in {city} during {season}. Style: {style}. Vibe: {vibe}. Lifestyle, cinematic light, no text.",
        f"Photorealistic fashion editorial inspired by {city}. Season: {season}. Style: {style}. Vibe: {vibe}. Clean composition, premium look, no text.",
    ]


def generate_images(prompts: List[str], image_model: str) -> List[Image.Image]:
    """
    config(response_modalities) 없이 단순 호출.
    """
    client = gemini_client()
    imgs: List[Image.Image] = []

    for p in prompts:
        resp = client.models.generate_content(model=image_model, contents=[p])

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
            continue

    return imgs


def text_mood_fallback(city: str, season: str, style: str, vibe: str, text_model: str) -> Dict[str, Any]:
    prompt = f"""
너는 크리에이티브 디렉터야. {city}/{season}/{style}/{vibe}로 무드보드 텍스트를 만들어.
반드시 JSON만 출력.

스키마:
{{
  "headline": "한 줄 컨셉",
  "keywords": ["키워드 8~12개"],
  "shot_list": ["샷1","샷2","샷3","샷4"]
}}
"""
    return call_json(prompt, model=text_model)


# =====================
# UI
# =====================
st.set_page_config(page_title="Tripfit", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 0.9rem; max-width: 1200px; }
.big-title { font-size: 2.1rem; font-weight: 900; letter-spacing: -0.02em; }
.subtle { color: rgba(0,0,0,0.55); }
.hero {
  background: linear-gradient(135deg, rgba(255,255,255,0.82), rgba(255,255,255,0.58));
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 26px;
  padding: 18px;
  box-shadow: 0 12px 44px rgba(0,0,0,0.07);
}
.card {
  background: rgba(255,255,255,0.80);
  border: 1px solid rgba(0,0,0,0.07);
  border-radius: 20px;
  padding: 14px 14px;
  box-shadow: 0 8px 30px rgba(0,0,0,0.05);
}
.hr { height: 1px; background: rgba(0,0,0,0.08); margin: 12px 0; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">Tripfit 🍌</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">무드보드가 먼저 · 이미지가 막히면 원인을 화면에 보여줄게</div>', unsafe_allow_html=True)

st.session_state.setdefault("mood_imgs", [])
st.session_state.setdefault("mood_text", None)
st.session_state.setdefault("weather", [])
st.session_state.setdefault("place", "")

with st.sidebar:
    st.markdown("### 🔑 Gemini API Key")
    st.text_input("API Key", type="password", key="api_key_input", placeholder="paste AI Studio key")
    st.caption("✅ ready" if get_api_key() else "키가 필요해요")

    st.markdown("---")
    city = st.text_input("City", value="Tokyo")
    c1, c2 = st.columns(2)
    start = c1.date_input("From", value=date.today() + timedelta(days=7))
    end = c2.date_input("To", value=date.today() + timedelta(days=10))

    style = st.selectbox("Style", ["미니멀", "빈티지", "스트릿", "클래식", "러블리", "시티보이/시티걸", "고프코어", "기타"])
    vibe = st.text_input("Vibe", value="clean, chic, city walk, travel street style")
    season = st.text_input("Season", value="current season")

    st.markdown("---")
    text_model = st.text_input("Text model", value=TEXT_MODEL_DEFAULT)
    image_model = st.text_input("Image model", value=IMAGE_MODEL_DEFAULT)

st.markdown('<div class="hero">', unsafe_allow_html=True)
h1, h2 = st.columns([1.4, 1])
with h1:
    st.markdown("### 🍌 Moodboard")
    st.markdown('<div class="subtle">4컷을 크게 보여줄게. 실패하면 상태코드를 바로 노출.</div>', unsafe_allow_html=True)
with h2:
    test = st.button("Quick Test (1 image)", use_container_width=True)
    gen = st.button("Generate (4 images)", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

def show_error(e: Exception):
    # Streamlit redaction 때문에 “상태코드”라도 최대한 보여주기
    status = getattr(e, "status_code", None)
    msg = getattr(e, "message", None) or (e.args[0] if getattr(e, "args", None) else "")
    st.error(f"Image API rejected. status={status}  detail={msg}")

    st.info(
        "이 경우는 보통 코드 문제가 아니라 **키/프로젝트 권한 또는 Billing** 문제야.\n"
        "- AI Studio에서 같은 키로 이미지 모델이 되는지 확인\n"
        "- 안 되면 해당 프로젝트는 이미지 생성이 막힌 상태"
    )

# Quick test
if test:
    if not get_api_key():
        st.error("키부터 입력해줘.")
    else:
        try:
            prompt = f"Photorealistic fashion photo in {city}. Style: {style}. Vibe: {vibe}. Full body, no text."
            with st.spinner("testing…"):
                imgs = generate_images([prompt], image_model=image_model)
            if imgs:
                st.session_state.mood_imgs = imgs
                st.session_state.mood_text = None
            else:
                st.warning("이미지 0장 반환. (권한/모델/쿼터 가능)")
        except genai_errors.ClientError as e:
            show_error(e)
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# Generate 4
if gen:
    if not get_api_key():
        st.error("키부터 입력해줘.")
    else:
        # 날씨 갱신(직관 카드)
        try:
            geo = geocode_city(city)
            if geo:
                lat, lon, nm, country = geo
                st.session_state.place = f"{nm}, {country}"
                f = forecast_daily(lat, lon, start, end)
                st.session_state.weather = weather_cards(f)
        except Exception:
            pass

        prompts = build_prompts(city, season, style, vibe)

        try:
            with st.spinner("creating…"):
                imgs = generate_images(prompts, image_model=image_model)
            if imgs:
                st.session_state.mood_imgs = imgs
                st.session_state.mood_text = None
            else:
                st.warning("이미지가 0장 반환됐어. 텍스트 무드보드로 대체할게.")
                st.session_state.mood_imgs = []
                st.session_state.mood_text = text_mood_fallback(city, season, style, vibe, text_model=text_model)

        except genai_errors.ClientError as e:
            show_error(e)
            st.session_state.mood_imgs = []
            st.session_state.mood_text = text_mood_fallback(city, season, style, vibe, text_model=text_model)

        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.session_state.mood_imgs = []
            st.session_state.mood_text = text_mood_fallback(city, season, style, vibe, text_model=text_model)

# Render moodboard (big 2x2)
imgs = st.session_state.mood_imgs
if imgs:
    c1, c2 = st.columns(2)
    c1.image(imgs[0], use_container_width=True)
    if len(imgs) > 1:
        c2.image(imgs[1], use_container_width=True)
    if len(imgs) > 2:
        c1.image(imgs[2], use_container_width=True)
    if len(imgs) > 3:
        c2.image(imgs[3], use_container_width=True)
else:
    t = st.session_state.mood_text
    if t:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"### {t.get('headline','')}")
        st.write(" · ".join(t.get("keywords", [])))
        st.markdown("**shot list**")
        for s in t.get("shot_list", []):
            st.write(f"- {s}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="subtle" style="margin-top: 12px;">Generate를 눌러 무드보드를 만들어봐.</div>', unsafe_allow_html=True)

# Weather (simple cards)
if st.session_state.weather:
    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🌦️ Weather")
    if st.session_state.place:
        st.caption(st.session_state.place)
    cols = st.columns(min(5, len(st.session_state.weather)))
    for i, w in enumerate(st.session_state.weather[:5]):
        with cols[i]:
            st.markdown(
                f"**{w['date']}** {w['icon']}  \n"
                f"**{w['tmin']}° ~ {w['tmax']}°**  \n"
                f"<span class='subtle'>☔ {w['pop']}% · 💨 {w['wind']}km/h</span>",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)
