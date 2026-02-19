import json
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types

# ========= Models =========
TEXT_MODEL = "gemini-2.5-flash"
IMAGE_MODEL = "gemini-2.5-flash-image"  # Nano Banana 계열(네이티브 이미지 생성)

# ========= Helpers =========
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


def get_api_key() -> Optional[str]:
    # 화면 입력(세션에만 저장됨)
    key = st.session_state.get("api_key_input")
    if key and key.strip():
        return key.strip()

    # (선택) secrets 지원
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]

    return None


def gemini_client() -> genai.Client:
    key = get_api_key()
    if not key:
        raise RuntimeError("Gemini API Key가 필요합니다.")
    return genai.Client(api_key=key)


def call_gemini_text(prompt: str, temperature: float = 0.7) -> str:
    client = gemini_client()
    resp = client.models.generate_content(
        model=TEXT_MODEL,
        contents=[prompt],
        config=types.GenerateContentConfig(temperature=temperature),
    )
    return getattr(resp, "text", "") or ""


def call_gemini_json(prompt: str, retries: int = 2) -> Dict[str, Any]:
    rule = "반드시 유효한 JSON만 출력. 다른 텍스트/설명/마크다운/코드펜스 금지."
    last = None
    for _ in range(retries + 1):
        try:
            txt = call_gemini_text(rule + "\n" + prompt, temperature=0.4)
            return _safe_json_loads(txt)
        except Exception as e:
            last = e
    raise RuntimeError(f"Gemini JSON 파싱 실패: {last}")


# ========= Weather (Open-Meteo) =========
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


def _temp_label(tmin: float, tmax: float) -> str:
    avg = (tmin + tmax) / 2
    if avg <= 0:
        return "🧊 매우 추움"
    if avg <= 8:
        return "🧥 쌀쌀"
    if avg <= 16:
        return "🧶 선선"
    if avg <= 24:
        return "👕 따뜻"
    return "🥵 더움"


def _rain_icon(pop: float) -> str:
    if pop >= 70:
        return "🌧️"
    if pop >= 40:
        return "🌦️"
    if pop >= 20:
        return "☁️"
    return "☀️"


def format_weather_cards(f: Dict[str, Any]) -> List[Dict[str, Any]]:
    d = f.get("daily", {}) or {}
    times = d.get("time", []) or []
    tmax = d.get("temperature_2m_max", []) or []
    tmin = d.get("temperature_2m_min", []) or []
    pop = d.get("precipitation_probability_max", []) or []
    wind = d.get("windspeed_10m_max", []) or []

    cards = []
    for i in range(len(times)):
        tmin_i = float(tmin[i])
        tmax_i = float(tmax[i])
        pop_i = float(pop[i])
        wind_i = float(wind[i])
        cards.append(
            {
                "date": times[i],
                "icon": _rain_icon(pop_i),
                "temp": f"{int(round(tmin_i))}° ~ {int(round(tmax_i))}°",
                "feel": _temp_label(tmin_i, tmax_i),
                "rain": f"강수 {int(round(pop_i))}%",
                "wind": f"바람 {int(round(wind_i))}km/h",
            }
        )
    return cards


# ========= Moodboard (Nano Banana) =========
@st.cache_data(show_spinner=False)
def moodboard_prompts(city: str, season: str, style: str, vibe: str) -> List[str]:
    # 프롬프트를 다양하게 잡아 "비슷비슷한 사진" 덜 나오게
    return [
        f"Photorealistic street-style fashion photo in {city} during {season}. Style: {style}. Vibe: {vibe}. Full body, natural light, no text, high detail.",
        f"Photorealistic outfit flat-lay on a warm neutral background. Destination: {city}, season: {season}. Style: {style}. Include 7-9 items, no text, high detail.",
        f"Photorealistic candid travel moment in {city} during {season}. Style: {style}. Vibe: {vibe}. Lifestyle, cinematic light, no text.",
        f"Photorealistic fashion editorial inspired by {city}. Season: {season}. Style: {style}. Vibe: {vibe}. Clean composition, premium look, no text.",
    ]


def generate_moodboard_images(prompts: List[str]) -> List[Image.Image]:
    client = gemini_client()
    imgs: List[Image.Image] = []

    for p in prompts:
        resp = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=[p],
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )

        parts = []
        if hasattr(resp, "parts") and resp.parts:
            parts = resp.parts
        elif hasattr(resp, "candidates") and resp.candidates:
            parts = resp.candidates[0].content.parts

        got = False
        for part in parts:
            if getattr(part, "inline_data", None) is not None:
                imgs.append(part.as_image())
                got = True
                break

        if not got:
            # 한 장 실패해도 나머지는 진행
            continue

    return imgs


# ========= UI =========
st.set_page_config(page_title="Tripfit", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 1.0rem; }
.big-title { font-size: 2.1rem; font-weight: 900; letter-spacing: -0.02em; }
.subtle { color: rgba(0,0,0,0.55); }

.card {
  background: rgba(255,255,255,0.78);
  border: 1px solid rgba(0,0,0,0.07);
  border-radius: 20px;
  padding: 16px 16px;
  box-shadow: 0 8px 30px rgba(0,0,0,0.05);
}

.pill {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(0,0,0,0.06);
  margin-right: 6px;
  margin-bottom: 6px;
  font-size: 0.85rem;
}

.hr { height: 1px; background: rgba(0,0,0,0.08); margin: 12px 0; }

.mood-wrap {
  background: linear-gradient(135deg, rgba(255,255,255,0.78), rgba(255,255,255,0.55));
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 24px;
  padding: 18px;
  box-shadow: 0 10px 40px rgba(0,0,0,0.06);
}

.mood-title { font-size: 1.25rem; font-weight: 800; margin-bottom: 4px; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">Tripfit ✈️👗</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">destination mood → outfit ideas → moodboard</div>', unsafe_allow_html=True)

# state
st.session_state.setdefault("outfits", [])
st.session_state.setdefault("weather_cards", [])
st.session_state.setdefault("weather_place", "")
st.session_state.setdefault("mood_imgs", [])
st.session_state.setdefault("mood_seed", 0)  # 버튼 클릭마다 값 증가시켜 rerun 안정화

# Sidebar (minimal)
with st.sidebar:
    st.markdown("### 🔑 Gemini Key")
    st.text_input(
        "API Key",
        type="password",
        key="api_key_input",
        placeholder="paste here",
        help="세션에만 저장돼요(새로고침하면 사라짐).",
    )
    st.caption("✅ ready" if get_api_key() else "키가 필요해요")

    st.markdown("---")
    st.markdown("### 🌍 Trip")
    destination = st.text_input("City", value="Tokyo")
    c1, c2 = st.columns(2)
    start_date = c1.date_input("From", value=date.today() + timedelta(days=7))
    end_date = c2.date_input("To", value=date.today() + timedelta(days=10))

    st.markdown("### ✨ Taste")
    style = st.selectbox("Style", ["미니멀", "빈티지", "스트릿", "클래식", "러블리", "시티보이/시티걸", "고프코어", "기타"])
    vibe = st.text_input("Vibe", value="clean, chic, city walk, travel street style")
    season_hint = st.text_input("Season (optional)", value="")

# ========= Top: Moodboard (Hero) =========
st.markdown('<div class="mood-wrap">', unsafe_allow_html=True)
colA, colB = st.columns([1.3, 1])
with colA:
    st.markdown('<div class="mood-title">🍌 Moodboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">핵심 기능 · 4컷으로 분위기를 먼저 잡자</div>', unsafe_allow_html=True)

    season_for_image = st.text_input(
        "Season for images",
        value=season_hint.strip() if season_hint.strip() else "current season",
        key="season_for_image",
    )

with colB:
    st.write("")
    st.write("")
    gen_mb = st.button("Generate Moodboard", type="primary", use_container_width=True)

if gen_mb:
    if not get_api_key():
        st.error("Gemini API Key를 먼저 입력해줘.")
    else:
        st.session_state.mood_seed += 1  # rerun 안정화(키 충돌 방지 목적)
        prompts = moodboard_prompts(destination, season_for_image, style, vibe)
        with st.spinner("creating…"):
            imgs = generate_moodboard_images(prompts)
        st.session_state.mood_imgs = imgs

# Moodboard gallery (big)
imgs = st.session_state.mood_imgs
if imgs:
    g1, g2 = st.columns(2)
    # 2x2 크게
    if len(imgs) >= 1:
        g1.image(imgs[0], use_container_width=True)
    if len(imgs) >= 2:
        g2.image(imgs[1], use_container_width=True)
    if len(imgs) >= 3:
        g1.image(imgs[2], use_container_width=True)
    if len(imgs) >= 4:
        g2.image(imgs[3], use_container_width=True)
else:
    st.markdown('<div class="subtle">아직 이미지가 없어요. 버튼을 눌러 4컷 무드를 만들어봐.</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.write("")

# ========= Lower: Weather + Outfit =========
left, right = st.columns([1, 1.2])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🌦️ Weather")
    wbtn = st.button("Update weather", use_container_width=True)

    if wbtn:
        geo = geocode_city(destination)
        if not geo:
            st.error("도시를 찾지 못했어. 영문 도시명으로도 시도해줘.")
        else:
            lat, lon, city_name, country = geo
            f = forecast_daily(lat, lon, start_date, end_date)
            st.session_state.weather_cards = format_weather_cards(f)
            st.session_state.weather_place = f"{city_name}, {country}"

    if st.session_state.weather_cards:
        st.caption(st.session_state.weather_place)
        for c in st.session_state.weather_cards:
            st.markdown(
                f"**{c['date']}**  {c['icon']}  **{c['temp']}**  · {c['feel']}  \n"
                f"<span class='subtle'>{c['rain']} · {c['wind']}</span>",
                unsafe_allow_html=True,
            )
            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    else:
        st.markdown("<span class='subtle'>Update weather를 누르면 여행 기간 예보가 카드로 보여.</span>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 👗 Outfit ideas")
    st.markdown("<span class='subtle'>무드보드 느낌을 유지하면서, 날씨를 반영한 룩 3개.</span>", unsafe_allow_html=True)
    obtn = st.button("Generate outfits", type="primary", use_container_width=True)

    if obtn:
        if not get_api_key():
            st.error("Gemini API Key를 먼저 입력해줘.")
        else:
            # 날씨 카드가 없으면 자동 업데이트
            if not st.session_state.weather_cards:
                geo = geocode_city(destination)
                if geo:
                    lat, lon, city_name, country = geo
                    f = forecast_daily(lat, lon, start_date, end_date)
                    st.session_state.weather_cards = format_weather_cards(f)
                    st.session_state.weather_place = f"{city_name}, {country}"

            weather_lines = []
            for c in st.session_state.weather_cards[:7]:
                weather_lines.append(f"{c['date']}: {c['temp']}, {c['rain']}, {c['wind']}")
            weather_text = "\n".join(weather_lines) if weather_lines else "날씨 정보 없음"

            prompt = f"""
너는 여행 스타일리스트야.

[입력]
- 도시: {destination}
- 기간: {start_date.isoformat()} ~ {end_date.isoformat()}
- 스타일: {style}
- 무드: {vibe}
- 시즌 힌트: {season_hint if season_hint.strip() else "없음"}
- 날씨(요약):
{weather_text}

[출력 JSON 스키마]
{{
  "outfits": [
    {{
      "title": "룩 이름(감성적으로)",
      "mood_tags": ["tag","tag"],
      "scenario": "언제 입는지(짧게)",
      "fit_and_color": "핏/컬러 한 줄",
      "items": ["아이템1", "아이템2", "아이템3", "아이템4", "아이템5"],
      "layering_tip": "날씨 대응 팁 1~2문장"
    }}
  ]
}}

[규칙]
- 반드시 JSON만 출력.
- items는 실제 의류 품목으로.
- 날씨가 비/바람/추움이면 대응 아이템 포함.
"""
            data = call_gemini_json(prompt)
            st.session_state.outfits = data.get("outfits", []) or []

    if st.session_state.outfits:
        for o in st.session_state.outfits:
            st.markdown(f"#### {o.get('title','')}")
            tags = o.get("mood_tags", []) or []
            if tags:
                st.markdown("".join([f"<span class='pill'>{t}</span>" for t in tags]), unsafe_allow_html=True)
            st.markdown(f"**{o.get('scenario','')}**")
            st.caption(o.get("fit_and_color", ""))

            items = o.get("items", []) or []
            if items:
                st.markdown("".join([f"<span class='pill'>{it}</span>" for it in items]), unsafe_allow_html=True)

            st.caption(o.get("layering_tip", ""))
            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    else:
        st.markdown("<span class='subtle'>Generate outfits를 누르면 룩 카드가 생성돼.</span>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
