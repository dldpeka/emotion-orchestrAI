import streamlit as st
import pandas as pd
from typing import TypedDict, List, Dict, Any, Optional
from typing_extensions import NotRequired
import re
from datetime import datetime
from collections import Counter
import json
import os

# LangGraph
from langgraph.graph import StateGraph, END

# HuggingFace
from transformers import pipeline

# 문장 분리
from kss import split_sentences

# OpenAI
from openai import OpenAI

# Tavily
from tavily import TavilyClient

# ============================================================================
# 페이지 설정
# ============================================================================
st.set_page_config(
    page_title="🎭 감정 분석 멀티에이전트",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 세션 상태 초기화
# ============================================================================
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'emotion_classifier' not in st.session_state:
    st.session_state.emotion_classifier = None
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# ============================================================================
# State 정의
# ============================================================================
class AppState(TypedDict, total=False):
    raw_input: str
    input_type: str
    analysis_mode: str
    messages: List[Dict[str, Any]]
    text: str
    required_agents: List[str]
    emotion_df: NotRequired[pd.DataFrame]
    agg_result: Dict[str, Any]
    insight_text: str
    content_query: str
    content_recos: List[Dict[str, str]]
    completed_agents: List[str]

# ============================================================================
# 모델 로딩 함수
# ============================================================================
@st.cache_resource
def load_emotion_model():
    """한국어 감정 분석 모델 로딩"""
    try:
        classifier = pipeline(
            "text-classification",
            model="Seonghaa/korean-emotion-classifier-roberta",
            top_k=3
        )
        return classifier
    except Exception as e:
        st.error(f"감정 분석 모델 로딩 실패: {e}")
        return None

def init_openai_client(api_key: str):
    """OpenAI 클라이언트 초기화"""
    if api_key:
        return OpenAI(api_key=api_key)
    return None

# ============================================================================
# 유틸리티 함수들
# ============================================================================
def parse_kakao_txt(text: str) -> List[Dict[str, Any]]:
    """카카오톡 텍스트 파싱"""
    messages = []
    pattern = r'(\d{4}년\s+\d{1,2}월\s+\d{1,2}일.*?),\s*(.+?)\s*:\s*(.+)'
    
    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        match = re.match(pattern, line)
        if match:
            datetime_str = match.group(1).strip()
            speaker = match.group(2).strip()
            msg_text = match.group(3).strip()
            
            messages.append({
                "speaker": speaker,
                "datetime": datetime_str,
                "text": msg_text,
                "emotions": []
            })
    
    return messages

def analyze_emotion(text: str, classifier) -> List[Dict[str, Any]]:
    """감정 분석"""
    if classifier is None:
        return [{"label": "중립", "score": 0.5}]
    
    try:
        results = classifier(text)[0]
        return results
    except Exception as e:
        return [{"label": "중립", "score": 0.5}]

def llm_orchestrator_decision(analysis_mode: str, text_preview: str, client) -> List[str]:
    """LLM 기반 에이전트 선택"""
    if analysis_mode == "emotion_only":
        return ["emotion"]
    elif analysis_mode == "insight_only":
        return ["emotion", "insight"]
    elif analysis_mode == "full":
        return ["emotion", "insight", "content"]
    
    if not client:
        return ["emotion", "insight"]
    
    try:
        prompt = f"""당신은 멀티에이전트 시스템의 Orchestrator입니다.
입력 데이터를 분석하여 필요한 에이전트들을 선택하세요.

**사용 가능한 에이전트:**
1. emotion - 감정 분석 에이전트
2. insight - 인사이트 생성 에이전트
3. content - 콘텐츠 추천 에이전트

**입력 데이터:**
{text_preview[:500]}...

필요한 에이전트 목록을 JSON 배열로만 답하세요 (예: ["emotion", "insight"]):"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 작업 분석 전문가입니다. JSON 배열만 반환하세요."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        result_text = response.choices[0].message.content.strip()
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        required = json.loads(result_text)
        return required
    
    except Exception as e:
        st.warning(f"Orchestrator 판단 실패: {e}, 기본값 사용")
        return ["emotion", "insight"]

def generate_llm_insight(agg_result: Dict[str, Any], emotion_df: pd.DataFrame, 
                        speaker_analysis: Dict[str, Any], client) -> str:
    """LLM 기반 인사이트 생성"""
    if not client:
        return "⚠️ OpenAI API 키가 필요합니다."
    
    try:
        emotion_summary = emotion_df.head(20).to_string()
        
        prompt = f"""다음은 대화 참여자들의 감정 분석 결과입니다:

📊 전체 집계:
{json.dumps(agg_result, ensure_ascii=False, indent=2)}

👥 화자별 분석:
{json.dumps(speaker_analysis, ensure_ascii=False, indent=2)}

📝 메시지별 상세 데이터 (샘플):
{emotion_summary}

위 데이터를 바탕으로 다음 내용을 포함한 분석 리포트를 작성해주세요:

1. **전반적인 감정 패턴 분석**
2. **화자별 감정 특성 및 관계 역학 분석**
3. **시간 흐름에 따른 변화**
4. **심리적 조언 및 관계 개선 제안**

따뜻하고 공감적인 톤으로 작성해주시고, 이모지를 적절히 활용해주세요."""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 감정 분석 및 관계 상담 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"⚠️ LLM 인사이트 생성 오류: {e}"

def search_with_tavily(query: str, api_key: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Tavily 검색"""
    if not api_key:
        return []
    
    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=max_results
        )
        
        results = []
        for item in response.get('results', []):
            url = item.get('url', '')
            if 'youtube.com' in url or 'youtu.be' in url:
                content_type = "video"
            elif any(domain in url for domain in ['news', 'article', 'blog']):
                content_type = "article"
            else:
                content_type = "content"
            
            results.append({
                "type": content_type,
                "title": item.get('title', 'No title'),
                "url": url,
                "snippet": item.get('content', '')[:150]
            })
        
        return results
    
    except Exception as e:
        st.error(f"Tavily 검색 오류: {e}")
        return []

# ============================================================================
# 에이전트 함수들
# ============================================================================
def aggregator_agent(state: AppState, classifier, openai_client) -> AppState:
    """전처리 및 Orchestrator"""
    raw_input = state.get("raw_input", "")
    analysis_mode = state.get("analysis_mode", "auto")
    
    text = raw_input
    messages = parse_kakao_txt(text)
    
    if not messages:
        sentences = split_sentences(text)
        messages = [
            {
                "speaker": "Unknown",
                "datetime": datetime.now().strftime("%Y년 %m월 %d일"),
                "text": sent,
                "emotions": []
            }
            for sent in sentences if sent.strip()
        ]
    
    required_agents = llm_orchestrator_decision(analysis_mode, text, openai_client)
    
    state["text"] = text
    state["messages"] = messages
    state["required_agents"] = required_agents
    state["completed_agents"] = []
    
    return state

def emotion_agent(state: AppState, classifier) -> AppState:
    """감정 분석"""
    messages = state.get("messages", [])
    
    for msg in messages:
        text = msg.get("text", "")
        emotions = analyze_emotion(text, classifier)
        msg["emotions"] = emotions
    
    rows = []
    for msg in messages:
        speaker = msg.get("speaker", "Unknown")
        datetime_str = msg.get("datetime", "")
        text = msg.get("text", "")
        emotions = msg.get("emotions", [])
        
        if emotions:
            top_emotion = max(emotions, key=lambda x: x["score"])
            emotion_label = top_emotion["label"]
            emotion_score = top_emotion["score"]
        else:
            emotion_label = "중립"
            emotion_score = 0.5
        
        rows.append({
            "speaker": speaker,
            "datetime": datetime_str,
            "text": text,
            "emotion": emotion_label,
            "score": emotion_score,
            "text_length": len(text)
        })
    
    emotion_df = pd.DataFrame(rows)
    
    total_msgs = len(messages)
    emotion_counts = Counter([row["emotion"] for row in rows])
    emotion_ratios = {k: v/total_msgs for k, v in emotion_counts.items()}
    dominant_label = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "중립"
    
    speaker_analysis = {}
    speakers = emotion_df["speaker"].unique()
    
    for speaker in speakers:
        speaker_msgs = emotion_df[emotion_df["speaker"] == speaker]
        speaker_emotions = Counter(speaker_msgs["emotion"].tolist())
        speaker_dominant = max(speaker_emotions, key=speaker_emotions.get) if speaker_emotions else "중립"
        avg_score = speaker_msgs["score"].mean()
        
        speaker_analysis[speaker] = {
            "message_count": len(speaker_msgs),
            "dominant_emotion": speaker_dominant,
            "emotion_distribution": dict(speaker_emotions),
            "avg_score": avg_score
        }
    
    agg_result = {
        "total_msgs": total_msgs,
        "counts": dict(emotion_counts),
        "ratios": emotion_ratios,
        "dominant_label": dominant_label,
        "speaker_analysis": speaker_analysis
    }
    
    state["messages"] = messages
    state["emotion_df"] = emotion_df
    state["agg_result"] = agg_result
    state["completed_agents"] = state.get("completed_agents", []) + ["emotion"]
    
    return state

def insight_agent(state: AppState, openai_client) -> AppState:
    """인사이트 생성"""
    agg_result = state.get("agg_result", {})
    emotion_df = state.get("emotion_df", pd.DataFrame())
    speaker_analysis = agg_result.get("speaker_analysis", {})
    
    insight_text = generate_llm_insight(agg_result, emotion_df, speaker_analysis, openai_client)
    
    dominant = agg_result.get("dominant_label", "중립")
    dominant_lower = dominant.lower()
    
    if any(word in dominant_lower for word in ["슬픔", "우울", "sad", "부정"]):
        content_query = "우울 슬픔 감정 관리 심리 상담"
    elif any(word in dominant_lower for word in ["불안", "걱정", "anxious"]):
        content_query = "불안 걱정 스트레스 해소 마음챙김"
    elif any(word in dominant_lower for word in ["분노", "화", "angry"]):
        content_query = "분노 조절 갈등 해결 의사소통"
    elif any(word in dominant_lower for word in ["기쁨", "행복", "긍정", "positive", "happy"]):
        content_query = "행복 긍정 감정 유지 관계 개선"
    else:
        content_query = "감정 관리 심리 건강 자기계발"
    
    state["insight_text"] = insight_text
    state["content_query"] = content_query
    state["completed_agents"] = state.get("completed_agents", []) + ["insight"]
    
    return state

def content_recommender_agent(state: AppState, tavily_api_key: str) -> AppState:
    """콘텐츠 추천"""
    content_query = state.get("content_query", "감정 관리")
    content_recos = search_with_tavily(content_query, tavily_api_key, max_results=5)
    
    state["content_recos"] = content_recos
    state["completed_agents"] = state.get("completed_agents", []) + ["content"]
    
    return state

# ============================================================================
# 라우팅 함수들
# ============================================================================
def route_after_aggregator(state: AppState) -> str:
    required = state.get("required_agents", [])
    if "emotion" in required:
        return "emotion"
    elif "insight" in required:
        return "insight"
    elif "content" in required:
        return "content"
    else:
        return "end"

def route_after_emotion(state: AppState) -> str:
    required = state.get("required_agents", [])
    completed = state.get("completed_agents", [])
    remaining = [a for a in required if a not in completed]
    
    if "insight" in remaining:
        return "insight"
    elif "content" in remaining:
        return "content"
    else:
        return "end"

def route_after_insight(state: AppState) -> str:
    required = state.get("required_agents", [])
    completed = state.get("completed_agents", [])
    remaining = [a for a in required if a not in completed]
    
    if "content" in remaining:
        return "content"
    else:
        return "end"

# ============================================================================
# 분석 실행 함수
# ============================================================================
def run_analysis(text: str, analysis_mode: str, openai_key: str, tavily_key: str, classifier):
    """전체 분석 파이프라인 실행"""
    
    # OpenAI 클라이언트 초기화
    openai_client = init_openai_client(openai_key) if openai_key else None
    
    # 초기 상태 생성
    initial_state = {
        "raw_input": text,
        "analysis_mode": analysis_mode
    }
    
    # 1. Aggregator 실행
    with st.status("🎭 Aggregator 실행 중...", expanded=True) as status:
        st.write("입력 전처리 및 에이전트 선택 중...")
        state = aggregator_agent(initial_state, classifier, openai_client)
        st.write(f"✅ {len(state['messages'])}개 메시지 파싱 완료")
        st.write(f"🎯 실행할 에이전트: {state['required_agents']}")
        status.update(label="✅ Aggregator 완료", state="complete")
    
    required = state.get("required_agents", [])
    
    # 2. Emotion Agent 실행
    if "emotion" in required:
        with st.status("🟢 EmotionAgent 실행 중...", expanded=True) as status:
            st.write("감정 분석 중...")
            state = emotion_agent(state, classifier)
            st.write(f"✅ 감정 분석 완료 - 주요 감정: {state['agg_result']['dominant_label']}")
            status.update(label="✅ EmotionAgent 완료", state="complete")
    
    # 3. Insight Agent 실행
    if "insight" in required:
        with st.status("🟡 InsightAgent 실행 중...", expanded=True) as status:
            st.write("인사이트 생성 중...")
            state = insight_agent(state, openai_client)
            st.write("✅ 인사이트 생성 완료")
            st.write(f"🔍 콘텐츠 검색 키워드: {state['content_query']}")
            status.update(label="✅ InsightAgent 완료", state="complete")
    
    # 4. Content Agent 실행
    if "content" in required:
        with st.status("🔴 ContentAgent 실행 중...", expanded=True) as status:
            st.write("콘텐츠 추천 중...")
            state = content_recommender_agent(state, tavily_key)
            st.write(f"✅ {len(state.get('content_recos', []))}개 콘텐츠 추천 완료")
            status.update(label="✅ ContentAgent 완료", state="complete")
    
    return state

# ============================================================================
# Streamlit UI
# ============================================================================
def main():
    st.title("🎭 감정 분석 멀티에이전트 시스템")
    st.markdown("### LangGraph + HuggingFace + OpenAI")
    
    # 사이드바 - API 키 설정
    with st.sidebar:
        st.header("⚙️ 설정")
        
        openai_key = st.text_input("OpenAI API Key", type="password", 
                                   help="필수 - Orchestrator와 인사이트 생성에 사용됩니다")
        tavily_key = st.text_input("Tavily API Key", type="password",
                                   help="선택 - 콘텐츠 추천에 사용됩니다")
        
        st.divider()
        
        st.header("🎯 분석 모드")
        analysis_mode = st.radio(
            "모드 선택",
            ["auto", "full", "emotion_only", "insight_only"],
            help="""
            - auto: LLM이 자동으로 판단
            - full: 전체 분석 (감정 + 인사이트 + 콘텐츠)
            - emotion_only: 감정 분석만
            - insight_only: 관계 인사이트만
            """
        )
        
        st.divider()
        
        # 모델 로딩
        if not st.session_state.models_loaded:
            if st.button("🚀 모델 로딩", use_container_width=True):
                with st.spinner("모델 로딩 중..."):
                    st.session_state.emotion_classifier = load_emotion_model()
                    st.session_state.models_loaded = True
                    st.success("✅ 모델 로딩 완료!")
                    st.rerun()
        else:
            st.success("✅ 모델 로딩됨")
    
    # 메인 영역
    if not st.session_state.models_loaded:
        st.warning("⚠️ 먼저 사이드바에서 모델을 로딩해주세요.")
        return
    
    # 입력 영역
    st.header("📝 대화 입력")
    
    # 입력 방식 선택
    input_method = st.radio(
        "입력 방식 선택",
        ["📁 파일 업로드", "✍️ 직접 입력"],
        horizontal=True
    )
    
    input_text = ""
    
    if input_method == "📁 파일 업로드":
        st.info("💡 카카오톡 → 대화방 → 설정(≡) → '대화 내보내기' → TXT 파일 저장")
        
        uploaded_file = st.file_uploader(
            "카카오톡 대화 내보내기 파일 (.txt)",
            type=['txt'],
            help="카카오톡에서 내보낸 텍스트 파일을 업로드하세요"
        )
        
        if uploaded_file is not None:
            # 파일 읽기 (인코딩 처리)
            try:
                input_text = uploaded_file.read().decode('utf-8')
                st.success(f"✅ 파일 로드 완료: {uploaded_file.name}")
                
                # 미리보기
                with st.expander("📄 파일 내용 미리보기 (처음 10줄)"):
                    preview_lines = input_text.split('\n')[:10]
                    st.text('\n'.join(preview_lines))
                    if len(input_text.split('\n')) > 10:
                        st.caption(f"... 외 {len(input_text.split('\n')) - 10}줄")
            except UnicodeDecodeError:
                try:
                    uploaded_file.seek(0)
                    input_text = uploaded_file.read().decode('cp949')
                    st.success(f"✅ 파일 로드 완료: {uploaded_file.name} (CP949 인코딩)")
                except:
                    st.error("❌ 파일 인코딩 오류. UTF-8 또는 ANSI 형식의 파일을 사용해주세요.")
    
    else:  # 직접 입력
        # 샘플 데이터
        sample_text = """2024년 12월 11일 오후 2:30, 김철수 : 오늘 정말 힘든 하루였어
2024년 12월 11일 오후 2:31, 이영희 : 무슨 일 있었어?
2024년 12월 11일 오후 2:32, 김철수 : 회사에서 일이 너무 많아서 스트레스 받아
2024년 12월 11일 오후 2:33, 이영희 : 힘들겠다 ㅠㅠ 너무 걱정되네
2024년 12월 11일 오후 2:35, 김철수 : 불안하고 우울해... 어떻게 해야 할지 모르겠어"""
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("📋 샘플 데이터", use_container_width=True):
                st.session_state.input_text = sample_text
        
        input_text = st.text_area(
            "카카오톡 대화 또는 일반 텍스트를 입력하세요",
            value=st.session_state.get('input_text', ''),
            height=250,
            help="카카오톡 형식: YYYY년 MM월 DD일 시간, 이름 : 메시지"
        )
    
    # 분석 실행
    if st.button("🚀 분석 시작", type="primary", use_container_width=True):
        if not input_text.strip():
            st.error("입력 텍스트를 입력해주세요.")
            return
        
        if not openai_key and analysis_mode in ["auto", "full", "insight_only"]:
            st.error("OpenAI API 키가 필요합니다.")
            return
        
        # 분석 실행
        result = run_analysis(
            input_text, 
            analysis_mode, 
            openai_key, 
            tavily_key,
            st.session_state.emotion_classifier
        )
        
        st.session_state.analysis_result = result
        st.success("✅ 분석 완료!")
    
    # 결과 표시
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        
        st.divider()
        st.header("📊 분석 결과")
        
        # 실행된 에이전트 표시
        completed = result.get("completed_agents", [])
        st.info(f"🎯 실행된 에이전트: {', '.join(completed)}")
        
        # 탭으로 결과 구분
        tabs = []
        if "emotion" in completed:
            tabs.append("📈 감정 분석")
        if "insight" in completed:
            tabs.append("💡 인사이트")
        if "content" in completed:
            tabs.append("🎬 콘텐츠 추천")
        
        if tabs:
            tab_objects = st.tabs(tabs)
            tab_idx = 0
            
            # 감정 분석 탭
            if "emotion" in completed:
                with tab_objects[tab_idx]:
                    agg = result.get("agg_result", {})
                    emotion_df = result.get("emotion_df", pd.DataFrame())
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("전체 메시지", agg.get("total_msgs", 0))
                    with col2:
                        st.metric("주요 감정", agg.get("dominant_label", "중립"))
                    with col3:
                        dominant_ratio = agg.get("ratios", {}).get(agg.get("dominant_label", ""), 0)
                        st.metric("주요 감정 비율", f"{dominant_ratio*100:.1f}%")
                    
                    st.subheader("📊 감정 분포")
                    emotion_counts = agg.get("counts", {})
                    if emotion_counts:
                        chart_df = pd.DataFrame(list(emotion_counts.items()), 
                                               columns=["감정", "개수"])
                        st.bar_chart(chart_df.set_index("감정"))
                    
                    st.subheader("👥 화자별 분석")
                    speaker_analysis = agg.get("speaker_analysis", {})
                    for speaker, data in speaker_analysis.items():
                        with st.expander(f"**{speaker}** - {data['message_count']}개 메시지"):
                            st.write(f"**주요 감정:** {data['dominant_emotion']}")
                            st.write(f"**평균 점수:** {data['avg_score']:.3f}")
                            st.write(f"**감정 분포:** {data['emotion_distribution']}")
                    
                    st.subheader("📋 메시지별 상세")
                    st.dataframe(emotion_df, use_container_width=True)
                
                tab_idx += 1
            
            # 인사이트 탭
            if "insight" in completed:
                with tab_objects[tab_idx]:
                    insight_text = result.get("insight_text", "")
                    st.markdown(insight_text)
                    
                    st.divider()
                    st.info(f"🔍 **콘텐츠 검색 키워드:** {result.get('content_query', '')}")
                    st.caption("👆 이 키워드가 ContentAgent에게 전달되어 관련 콘텐츠를 추천합니다")
                
                tab_idx += 1
            
            # 콘텐츠 추천 탭
            if "content" in completed:
                with tab_objects[tab_idx]:
                    content_recos = result.get("content_recos", [])
                    
                    if content_recos:
                        st.write(f"**총 {len(content_recos)}개의 추천 콘텐츠**")
                        
                        for idx, item in enumerate(content_recos, 1):
                            with st.container():
                                col1, col2 = st.columns([1, 11])
                                
                                with col1:
                                    if item["type"] == "video":
                                        st.write("🎬")
                                    elif item["type"] == "article":
                                        st.write("📰")
                                    else:
                                        st.write("🔗")
                                
                                with col2:
                                    st.markdown(f"**{idx}. [{item['title']}]({item['url']})**")
                                    st.caption(item['snippet'])
                                
                                st.divider()
                    else:
                        st.warning("추천 콘텐츠가 없습니다. Tavily API 키를 확인해주세요.")

if __name__ == "__main__":
    main()