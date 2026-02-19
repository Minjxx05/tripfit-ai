import json
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from PIL import Image

from google import genai
from google.genai import types
from google.genai import errors as genai_errors


# ========= Defaults =========
DEFAULT_TEXT_MODEL = "gemini-2.5-flash"
DEFAULT_IMAGE_MODEL = "gemini-2.5-flash-image"  # Nano Banana (official doc example) :contentReference[oaicite:1]{index=1}

IMAGE_MODEL_OPTIONS = [
    "gemini-2.5-flash-image",
    "gemini-3-pro-image-preview",  # Android docs mention this image model preview :contentReference[oaicite:2]{index=2}
]


# ========= JSON helper =========
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


# ========= Key / client =========
def get_api_key() -> Optional[str]:
    key = st.session_state.get("api_key_input")
    if key and key.strip():
        return key.strip()
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return None


def gemini_client() -> genai.Client:
    key = get_api_key()
    if not key:
        raise RuntimeError("Gemini API Key가 필요합니다.")
    return genai.Client(api_key=key)


def call_gemini_text(prompt: str, model: str, temperature: float = 0.7) -> str:
    client = gemini_client()
    resp = client.models.generate_content(
        model=model,
        contents=[prompt],
        config=types.GenerateContentConfig(temperature=temperature),
    )
    return getattr(resp, "text", "") or ""


def call_gemini_json(prompt: str, model: str, retries: int = 2) -> Dict[str, Any]:
    rule = "반드시 유효한 JSON만 출력. 다른 텍스트/설명/마크다운/코드펜스 금지."
    last = None
    for _ in range(retries + 1):
        try:
            txt = call_gemini_text(rule + "\n" + prompt, model=model, temperature=0.4)
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


# ========= Moodboard prompts =========
@st.cache_data(show_spinner=False)
def build_mood_prompts(city: str, season: str, style: str, vibe: str) -> List[str]:
    # 4컷을 각기 다른 shot으로
    return [
        f"Photorealistic street-style fashion photo in {city} during {season}. Style: {style}. Vibe: {vibe}. Full body, natural light, no text, high detail.",
        f"Photorealistic outfit flat-lay on warm neutral background. Destination: {city}, season: {season}. Style: {style}. Include 7-9 items, no text, high detail.",
        f"Photorealistic candid travel moment in {city} during {season}. Style: {style}. Vibe: {vibe}. Lifestyle, cinematic light, no text.",
        f"Photorealistic fashion editorial inspired by {city}. Season: {season}. Style: {style}. Vibe: {vibe}. Clean composition, premium look, no text.",
    ]


def generate_moodboard_images(prompts: List[str], image_model: str) -> List[Image.Image]:
    """
    IMPORTANT:
    - 공식 문서 예시처럼 config(response_modalities) 없이 호출 (ClientError 회피용). :contentReference[oaicite:3]{index=3}
    """
    client = gemini_client()
    imgs: List[Image.Image] = []

    for p in prompts:
        resp = client.models.generate_content(
            model=image_model,
            contents=[p],
        )

        # 공식 문서 예시: response.parts에서 text/inline_data를 분기 :contentReference[oaicite:4]{index=4}
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


def generate_text_moodboard(city: str, season: str, style: str, vibe: str, text_model: str) -> Dict[str, Any]:
    prompt = f"""
너는 크리에이티브 디렉터야. {city} / {season} / {style} / {vibe}로 무드보드를 텍스트로 구성해줘.

반드시 JSON만 출력.

스키마:
{{
  "headline": "한 줄 컨셉",
  "keywords": ["키워드 8~12개"],
  "color_palette": ["#RRGGBB", "#RRGGBB", "#RRGGBB", "#RRGGBB", "#RRGGBB"],
  "shot_list": [
    "샷 아이디어 1(짧게)",
    "샷 아이디어 2",
    "샷 아이디어 3",
    "샷 아이디어 4"
  ]
}}
"""
    return call_gemini_json(prompt, model=text_model)


# ========= UI =========
st.set_page_config(page_title="Tripfit", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 0.9rem; }
.big-title { font-size: 2.15rem; font-weight: 900; letter-spacing: -0.02em; }
.subtle { color: rgba(0,0,0,0.55); }

.card {
  background: rgba(255,255,255,0.78);
  border: 1px solid rgba(0,0,0,0.07);
  border-radius: 20px;
  padding: 16px 16px;
  box-shadow: 0 8px 30px rgba(0,0,0,0.05);
}

.hr { height: 1px; background: rgba(0,0,0,0.08); margin: 12px 0; }

.mood-wrap {
  background: linear-gradient(135deg, rgba(255,255,255,0.82), rgba(255,255,255,0.58));
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 26px;
  padding: 18px;
  box-shadow: 0 12px 44px rgba(0,0,0,0.07);
}

.mood-title { font-size: 1.35rem; font-weight: 900; margin-bottom: 2px; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">Tripfit ✈️👗</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">moodboard first · then outfit ideas</div>', unsafe_allow_html=True)

st.session_state.setdefault("weather_cards", [])
st.session_state.setdefault("weather_place", "")
st.session_state.setdefault("mood_imgs", [])
st.session_state.setdefault("mood_text_board", None)
st.session_state.setdefault("outfits", [])

# Sidebar
with st.sidebar:
    st.markdown("### 🔑 Gemini Key")
    st.text_input("API Key", type="password", key="api_key_input", placeholder="paste here")
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

    st.markdown("---")
    text_model = st.text_input("Text model", value=DEFAULT_TEXT_MODEL)
    image_model = st.selectbox("Image model", IMAGE_MODEL_OPTIONS, index=0)

# ========= HERO: Moodboard =========
st.markdown('<div class="mood-wrap">', unsafe_allow_html=True)
colA, colB = st.columns([1.4, 1])

with colA:
    st.markdown('<div class="mood-title">🍌 Moodboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">핵심 기능 · 4컷을 크게</div>', unsafe_allow_html=True)
    season_for_image = st.text_input(
        "Season for images",
        value=season_hint.strip() if season_hint.strip() else "current season",
        key="season_for_image",
    )

with colB:
    st.write("")
    st.write("")
    gen_mb = st.button("Generate Moodboard", type="primary", use_container_width=True)
    st.caption("안되면 아래에 텍스트 무드보드로 자동 폴백")

if gen_mb:
    if not get_api_key():
        st.error("Gemini API Key를 먼저 입력해줘.")
    else:
        prompts = build_mood_prompts(destination, season_for_image, style, vibe)

        try:
            with st.spinner("creating images…"):
                imgs = generate_moodboard_images(prompts, image_model=image_model)
            if not imgs:
                raise RuntimeError("이미지 결과가 비어있어요.")
            st.session_state.mood_imgs = imgs
            st.session_state.mood_text_board = None

        except genai_errors.ClientError:
            # Streamlit Cloud에서는 원문이 redacted되므로, 사용자가 할 수 있는 체크만 안내
            st.warning(
                "이미지 생성 호출이 거절됐어요(ClientError). 아래 텍스트 무드보드를 대신 만들었어.\n\n"
                "체크 포인트:\n"
                "- AI Studio에서 발급한 **Gemini API Key**가 맞는지\n"
                "- 해당 키/프로젝트에서 **Image Generation 모델 사용 권한/결제(필요 시)**이 켜져 있는지\n"
                "- Image model을 바꿔서 재시도(사이드바에서 선택)"
            )
            st.session_state.mood_imgs = []
            with st.spinner("creating text moodboard…"):
                st.session_state.mood_text_board = generate_text_moodboard(
                    destination, season_for_image, style, vibe, text_model=text_model
                )

        except Exception as e:
            st.warning(f"이미지 생성이 실패했어요: {e}\n텍스트 무드보드로 전환합니다.")
            st.session_state.mood_imgs = []
            with st.spinner("creating text moodboard…"):
                st.session_state.mood_text_board = generate_text_moodboard(
                    destination, season_for_image, style, vibe, text_model=text_model
                )

# Render moodboard (big)
imgs = st.session_state.mood_imgs
if imgs:
    g1, g2 = st.columns(2)
    if len(imgs) >= 1:
        g1.image(imgs[0], use_container_width=True)
    if len(imgs) >= 2:
        g2.image(imgs[1], use_container_width=True)
    if len(imgs) >= 3:
        g1.image(imgs[2], use_container_width=True)
    if len(imgs) >= 4:
        g2.image(imgs[3], use_container_width=True)

else:
    board = st.session_state.mood_text_board
    if board:
        st.markdown(f"### {board.get('headline','')}")
        cols = st.columns([1.2, 1])
        with cols[0]:
            st.markdown("**Keywords**")
            st.write(" · ".join(board.get("keywords", [])))
            st.markdown("**Shot list**")
            for s in board.get("shot_list", []):
                st.write(f"- {s}")
        with cols[1]:
            st.markdown("**Palette**")
            for c in board.get("color_palette", []):
                st.color_picker(c, value=c, disabled=True, key=f"pal_{c}")
    else:
        st.markdown('<div class="subtle">아직 무드보드가 없어요. 버튼을 눌러 만들어봐.</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.write("")

# ========= Weather + Outfit ideas =========
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
        st.markdown("<span class='subtle'>Update weather를 누르면 기간 예보가 카드로 보여.</span>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 👗 Outfit ideas")
    st.markdown("<span class='subtle'>무드보드 톤을 유지하면서 날씨 기반으로 3개.</span>", unsafe_allow_html=True)
    obtn = st.button("Generate outfits", type="primary", use_container_width=True)

    if obtn:
        if not get_api_key():
            st.error("Gemini API Key를 먼저 입력해줘.")
        else:
            # 날씨 없으면 자동 업데이트
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
            data = call_gemini_json(prompt, model=text_model)
            st.session_state.outfits = data.get("outfits", []) or []

    if st.session_state.outfits:
        for o in st.session_state.outfits:
            st.markdown(f"#### {o.get('title','')}")
            st.markdown(f"**{o.get('scenario','')}**")
            st.caption(o.get("fit_and_color", ""))

            items = o.get("items", []) or []
            if items:
                st.write(" · ".join(items))

            st.caption(o.get("layering_tip", ""))
            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    else:
        st.markdown("<span class='subtle'>Generate outfits를 누르면 룩 카드가 생성돼.</span>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
