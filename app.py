import os
import json
import base64
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types


# =========================
# Config
# =========================
TEXT_MODEL = "gemini-2.5-flash"          # 텍스트 생성용
IMAGE_MODEL = "gemini-2.5-flash-image"  # Nano Banana(이미지 생성) :contentReference[oaicite:4]{index=4}

APP_TITLE = "Tripfit ✈️👗"
APP_DESC = "목적지/날씨/스타일에 맞춘 코디 추천 + 가상 캐리어 패킹 + 무드보드 생성"


# =========================
# Utilities
# =========================
def get_api_key() -> Optional[str]:
    # 1) Streamlit secrets
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    # 2) env var
    return os.getenv("GEMINI_API_KEY")


def gemini_client() -> genai.Client:
    key = get_api_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY가 없습니다. Streamlit secrets 또는 환경변수로 설정해주세요.")
    return genai.Client(api_key=key)  # Google Gen AI SDK :contentReference[oaicite:5]{index=5}


def safe_json_loads(s: str) -> Dict[str, Any]:
    """
    모델이 가끔 ```json ... ``` 형태로 감싸서 주는 경우가 있어 방어적으로 파싱합니다.
    """
    s = s.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1] if s.count("```") >= 2 else s.strip("```")
    s = s.strip()
    # 혹시 앞뒤에 잡텍스트가 붙으면 가장 큰 JSON 덩어리만 추출 시도
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1:
        s = s[first:last + 1]
    return json.loads(s)


def call_gemini_text(prompt: str, temperature: float = 0.7) -> str:
    client = gemini_client()
    resp = client.models.generate_content(
        model=TEXT_MODEL,
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=temperature,
        ),
    )
    # google-genai 응답은 parts로 오기도 하고 text로 합쳐지기도 합니다.
    if getattr(resp, "text", None):
        return resp.text
    parts = []
    for p in getattr(resp, "parts", []) or []:
        if getattr(p, "text", None):
            parts.append(p.text)
    return "\n".join(parts).strip()


def call_gemini_structured(prompt: str, retries: int = 2) -> Dict[str, Any]:
    """
    JSON만 반환하도록 강하게 지시하고 파싱. 실패 시 재시도.
    """
    json_instruction = """
반드시 유효한 JSON만 출력하세요. 다른 텍스트/설명/마크다운/코드펜스 금지.
"""
    last_err = None
    for _ in range(retries + 1):
        try:
            text = call_gemini_text(json_instruction + "\n" + prompt, temperature=0.4)
            return safe_json_loads(text)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"JSON 파싱 실패: {last_err}")


def open_meteo_geocode(city: str) -> Optional[Tuple[float, float, str, str]]:
    """
    Open-Meteo Geocoding (무료)
    """
    url = "https://geocoding-api.open-meteo.com/v1/search"
    r = requests.get(url, params={"name": city, "count": 1, "language": "ko", "format": "json"}, timeout=20)
    r.raise_for_status()
    data = r.json()
    results = data.get("results") or []
    if not results:
        return None
    it = results[0]
    lat = it["latitude"]
    lon = it["longitude"]
    name = it.get("name", city)
    country = it.get("country", "")
    return lat, lon, name, country


def open_meteo_forecast(lat: float, lon: float, start: date, end: date) -> Dict[str, Any]:
    """
    Open-Meteo forecast (무료)
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "weathercode,temperature_2m_max,temperature_2m_min,precipitation_probability_max,windspeed_10m_max",
        "timezone": "auto",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def summarize_weather(forecast: Dict[str, Any]) -> str:
    daily = forecast.get("daily", {})
    dates = daily.get("time", [])
    tmax = daily.get("temperature_2m_max", [])
    tmin = daily.get("temperature_2m_min", [])
    pop = daily.get("precipitation_probability_max", [])
    wind = daily.get("windspeed_10m_max", [])

    if not dates:
        return "날씨 정보를 가져오지 못했습니다."

    lines = []
    for i in range(len(dates)):
        lines.append(
            f"- {dates[i]}: 최저 {tmin[i]}°C / 최고 {tmax[i]}°C, 강수확률 {pop[i]}%, 최대풍속 {wind[i]} km/h"
        )
    return "\n".join(lines)


def extract_items_from_outfits(outfits: List[Dict[str, Any]]) -> List[str]:
    items = []
    for o in outfits:
        for k in ["tops", "bottoms", "outerwear", "shoes", "accessories", "bags"]:
            for it in o.get(k, []) or []:
                if isinstance(it, str):
                    items.append(it.strip())
    # 중복 제거(순서 유지)
    seen = set()
    uniq = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def generate_moodboard_images(
    city: str,
    season: str,
    style: str,
    vibe: str,
    n: int = 4,
) -> List[Image.Image]:
    """
    Nano Banana(Gemini native image model)로 무드보드 이미지 생성
    """
    client = gemini_client()

    # 여러 장을 한 번에 뽑기보단, 4개 프롬프트를 각각 생성(실패 격리)
    prompts = [
        f"Create a photorealistic street-style fashion photo in {city} during {season}. Style: {style}. Vibe: {vibe}. Full body, natural light, influencer look, high detail, no text.",
        f"Create a photorealistic outfit flat-lay on a clean background. Destination: {city}, season: {season}. Style: {style}. Include 6-8 items (top, bottom, outerwear, shoes, bag, accessories). No text.",
        f"Create a photorealistic candid travel photo in {city} during {season}. Style: {style}. Vibe: {vibe}. Subject wearing a travel-appropriate outfit, realistic, no text.",
        f"Create a photorealistic fashion editorial shot inspired by {city} in {season}. Style: {style}. Vibe: {vibe}. Clean composition, no text.",
    ][:n]

    images: List[Image.Image] = []
    for p in prompts:
        resp = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=[p],
            # 일부 SDK 예시처럼 config를 생략해도 생성되지만,
            # 안정성을 위해 "TEXT/IMAGE" 둘 다 허용
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"]
            ),
        )

        # parts에서 inline image 찾기
        got = False
        parts = []
        # SDK 버전에 따라 candidates 구조가 있을 수 있어 방어적으로 접근
        if hasattr(resp, "parts") and resp.parts:
            parts = resp.parts
        elif hasattr(resp, "candidates") and resp.candidates:
            parts = resp.candidates[0].content.parts

        for part in parts:
            if getattr(part, "inline_data", None) is not None:
                img = part.as_image()
                images.append(img)
                got = True
                break
        if not got:
            # 이미지가 안 왔으면 스킵(앱은 계속 진행)
            continue

    return images


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(APP_DESC)

with st.expander("✅ PRD 기반 구현 범위(3대 기능)", expanded=False):
    st.markdown(
        """
- (1) **AI 맞춤 코디 제안**: 목적지/날씨/스타일 입력 → 3가지 코디 + 이유 생성  
- (2) **가상 캐리어 패킹**: 코디 아이템을 체크리스트로 관리  
- (3) **여행지 무드보드**: Nano Banana(이미지 모델)로 도시/계절/스타일 무드 이미지 생성
"""
    )

# Sidebar inputs
st.sidebar.header("여행 정보 입력")
destination = st.sidebar.text_input("목적지(도시명)", value="Paris")
col_d1, col_d2 = st.sidebar.columns(2)
start_date = col_d1.date_input("여행 시작일", value=date.today() + timedelta(days=7))
end_date = col_d2.date_input("여행 종료일", value=date.today() + timedelta(days=10))

gender = st.sidebar.selectbox("성별", ["여성", "남성", "기타/선택안함"])
style = st.sidebar.selectbox("스타일 취향", ["미니멀", "빈티지", "스트릿", "클래식", "러블리", "시티보이/시티걸", "고프코어", "기타"])
age_band = st.sidebar.selectbox("연령대", ["10대", "20대", "30대", "40대", "50대+"])
activities = st.sidebar.multiselect(
    "주요 일정(TPO)",
    ["박물관/미술관", "맛집/카페", "자연/트레킹", "야경/바", "쇼핑", "비즈니스/세미나", "테마파크"],
    default=["맛집/카페", "박물관/미술관"],
)
budget = st.sidebar.selectbox("예산 감도", ["가성비", "중간", "프리미엄"])
season_hint = st.sidebar.text_input("계절/체감(선택)", value="")

st.sidebar.divider()
st.sidebar.subheader("Gemini API 키 상태")
key_ok = bool(get_api_key())
st.sidebar.write("✅ 설정됨" if key_ok else "❌ 미설정")
st.sidebar.caption("Streamlit Cloud에서는 Secrets에 GEMINI_API_KEY를 등록하세요.")

# Main
tab1, tab2, tab3 = st.tabs(["1) 코디 추천", "2) 캐리어 패킹", "3) 무드보드 (Nano Banana)"])

# Session state
if "outfits" not in st.session_state:
    st.session_state.outfits = []
if "weather_text" not in st.session_state:
    st.session_state.weather_text = ""
if "packing" not in st.session_state:
    st.session_state.packing = []  # list[str]
if "packed" not in st.session_state:
    st.session_state.packed = set()


with tab1:
    st.subheader("AI 맞춤 코디 제안")
    st.write("목적지 좌표/날씨를 불러온 뒤, Gemini가 3가지 코디를 JSON으로 생성합니다.")

    run_btn = st.button("🧠 코디 생성하기", type="primary", use_container_width=True)

    if run_btn:
        if not key_ok:
            st.error("GEMINI_API_KEY가 필요합니다. Streamlit Secrets 또는 환경변수로 설정해주세요.")
        else:
            if end_date < start_date:
                st.error("종료일은 시작일보다 같거나 이후여야 합니다.")
            else:
                with st.spinner("도시 검색 및 날씨 불러오는 중..."):
                    geo = open_meteo_geocode(destination)
                    if not geo:
                        st.error("도시를 찾지 못했어요. 영문 도시명으로 다시 시도해보세요.")
                        st.stop()
                    lat, lon, city_name, country = geo

                    # Open-Meteo는 최대 기간 제한이 있을 수 있어, 14일 이상은 요약만
                    # (필요하면 여기에서 기간을 잘라서 호출)
                    forecast = open_meteo_forecast(lat, lon, start_date, end_date)
                    weather_text = summarize_weather(forecast)
                    st.session_state.weather_text = weather_text

                st.markdown("### 🌦️ 여행 기간 날씨(요약)")
                st.markdown(weather_text)

                with st.spinner("Gemini가 코디를 추천 중..."):
                    tpo = ", ".join(activities) if activities else "일반 여행"
                    season_line = season_hint.strip() if season_hint.strip() else "알 수 없음(날씨 기반 판단)"
                    prompt = f"""
너는 여행 코디 스타일리스트야. 아래 정보를 바탕으로 3가지 코디를 추천해줘.
반드시 아래 JSON 스키마를 지켜.

[입력]
- 목적지: {city_name}, {country}
- 기간: {start_date.isoformat()} ~ {end_date.isoformat()}
- 여행자: {age_band}, 성별 {gender}
- 스타일 취향: {style}
- 일정(TPO): {tpo}
- 예산 감도: {budget}
- 계절/체감 힌트: {season_line}
- 날씨 상세:
{weather_text}

[출력 JSON 스키마]
{{
  "outfits": [
    {{
      "title": "코디 이름(짧게)",
      "scenario": "언제/어디에 입는지(TPO)",
      "tops": ["..."],
      "bottoms": ["..."],
      "outerwear": ["..."],
      "shoes": ["..."],
      "bags": ["..."],
      "accessories": ["..."],
      "why": "날씨/스타일/활동 관점의 추천 이유(2~4문장)",
      "layering_tip": "레이어링/온도 대응 팁(1~2문장)"
    }}
  ],
  "general_tips": ["여행 코디 팁 3개"]
}}

[규칙]
- 브랜드명/가격 언급 금지.
- 너무 추상적인 단어 대신 실제 의류 품목으로 작성(예: '코트' OK, '예쁜 옷' NO).
- 날씨가 춥거나 비/바람이 있으면 대응 아이템(방풍/우산/방수 신발 등)을 포함.
"""
                    data = call_gemini_structured(prompt)
                    outfits = data.get("outfits", [])
                    st.session_state.outfits = outfits

                    # 패킹 리스트도 동기화
                    items = extract_items_from_outfits(outfits)
                    st.session_state.packing = items

                st.success("코디 생성 완료!")

    if st.session_state.outfits:
        st.markdown("### 👗 추천 코디 3가지")
        for i, o in enumerate(st.session_state.outfits, start=1):
            with st.container(border=True):
                st.markdown(f"#### {i}. {o.get('title','(제목 없음)')}")
                st.caption(o.get("scenario", ""))

                cols = st.columns(3)
                cols[0].markdown("**상의**\n" + "\n".join([f"- {x}" for x in (o.get("tops") or [])]))
                cols[0].markdown("**아우터**\n" + "\n".join([f"- {x}" for x in (o.get('outerwear') or [])]))

                cols[1].markdown("**하의**\n" + "\n".join([f"- {x}" for x in (o.get("bottoms") or [])]))
                cols[1].markdown("**신발**\n" + "\n".join([f"- {x}" for x in (o.get('shoes') or [])]))

                cols[2].markdown("**가방**\n" + "\n".join([f"- {x}" for x in (o.get("bags") or [])]))
                cols[2].markdown("**액세서리**\n" + "\n".join([f"- {x}" for x in (o.get('accessories') or [])]))

                st.markdown("**추천 이유**")
                st.write(o.get("why", ""))
                st.markdown("**레이어링/날씨 대응 팁**")
                st.write(o.get("layering_tip", ""))

        st.markdown("### ✅ 일반 팁")
        # general_tips가 없으면 표시 생략
        # (여기서는 call_gemini_structured에서 함께 받도록 했지만, 방어적으로)
        # tips는 마지막 실행의 data를 들고 있지 않으니, 간단히 텍스트 모델로 즉석 생성
        if st.button("여행 코디 팁 다시 생성"):
            t = call_gemini_text(
                f"{destination} 여행(스타일:{style}) 코디 일반 팁 3가지만 불릿으로 짧게 써줘.",
                temperature=0.6,
            )
            st.write(t)


with tab2:
    st.subheader("가상 캐리어 패킹(체크리스트)")

    if not st.session_state.packing:
        st.info("먼저 1) 코디 추천을 생성하면 아이템이 자동으로 들어옵니다.")
    else:
        st.write("추천 코디에서 추출한 아이템을 체크하면서 짐을 꾸릴 수 있어요.")

        # 추가 아이템 입력
        add_item = st.text_input("추가할 아이템(선택)", placeholder="예: 히트텍, 접이식 우산")
        if st.button("➕ 추가", use_container_width=True):
            if add_item.strip():
                if add_item.strip() not in st.session_state.packing:
                    st.session_state.packing.append(add_item.strip())
                else:
                    st.warning("이미 목록에 있어요.")

        st.divider()

        # 체크리스트 표시
        packed_now = set(st.session_state.packed)
        for item in st.session_state.packing:
            checked = item in packed_now
            new_val = st.checkbox(item, value=checked, key=f"pack_{item}")
            if new_val:
                packed_now.add(item)
            else:
                packed_now.discard(item)

        st.session_state.packed = packed_now

        st.divider()
        total = len(st.session_state.packing)
        done = len(st.session_state.packed)
        st.metric("패킹 진행률", f"{done}/{total}")

        # 간단 “구매/보완 제안” (실제 쇼핑몰 연동 대신 텍스트 추천)
        if st.button("🛍️ 부족 아이템 보완 제안 받기", type="secondary", use_container_width=True):
            missing = [x for x in st.session_state.packing if x not in st.session_state.packed]
            if not missing:
                st.success("이미 다 챙겼어요! 👍")
            else:
                prompt = f"""
너는 여행 짐 패킹 컨설턴트야.
목적지: {destination}
스타일: {style}
날씨:
{st.session_state.weather_text}

아직 안 챙긴 목록:
{missing}

1) 누락되면 여행에서 불편할 수 있는 상위 5개를 골라 중요도 순으로 설명
2) 대체 가능한 아이템/간단한 구매 기준(브랜드/가격 언급 금지)
불릿으로 간단히.
"""
                st.write(call_gemini_text(prompt, temperature=0.5))


with tab3:
    st.subheader("여행지 무드보드 (Nano Banana 이미지 생성)")
    st.write("도시/계절/스타일 무드를 반영한 이미지 4장을 생성합니다. (텍스트+이미지 멀티모달)")

    vibe = st.text_input("무드 키워드(선택)", value="clean, chic, travel street style")
    season_for_image = st.text_input("계절(이미지용)", value=season_hint if season_hint.strip() else "current season")

    gen_mb = st.button("🍌 무드보드 만들기", type="primary", use_container_width=True)
    if gen_mb:
        if not key_ok:
            st.error("GEMINI_API_KEY가 필요합니다. Streamlit Secrets 또는 환경변수로 설정해주세요.")
        else:
            with st.spinner("Nano Banana로 무드보드 생성 중..."):
                imgs = generate_moodboard_images(
                    city=destination,
                    season=season_for_image,
                    style=style,
                    vibe=vibe,
                    n=4,
                )
            if not imgs:
                st.warning("이미지 생성에 실패했어요. 잠시 후 다시 시도하거나 프롬프트를 바꿔보세요.")
            else:
                cols = st.columns(4)
                for i, im in enumerate(imgs):
                    cols[i % 4].image(im, use_container_width=True)

    st.caption("참고: Nano Banana는 Gemini의 네이티브 이미지 생성 기능(예: gemini-2.5-flash-image)입니다. :contentReference[oaicite:6]{index=6}")

