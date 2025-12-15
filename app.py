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
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'keybert_model' not in st.session_state:
    st.session_state.keybert_model = None

# ============================================================================
# State 정의
# ============================================================================
class AppState(TypedDict, total=False):
    # 입력 데이터
    raw_input: str
    text: str
    messages: List[Dict[str, Any]]
    
    # EmotionAgent 결과
    emotion_df: NotRequired[pd.DataFrame]
    emotion_summary: Dict[str, Any]
    
    # 병렬 실행 Agent 결과들
    insight_text: str  # InsightAgent
    statistical_summary: Dict[str, Any]  # SummaryAgent
    extracted_keywords: List[str]  # KeywordExtractorAgent
    
    # ContentAgent 결과
    content_query: str
    content_query_display: str 
    content_recos: List[Dict[str, str]]
    
    # Aggregator 최종 결과
    final_report: Dict[str, Any]
    
    # 진행 상황 추적
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


@st.cache_resource
def load_keybert_model():
    """KeyBERT 모델 로딩 - 가벼운 다국어 모델 사용"""
    try:
        from keybert import KeyBERT
        kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
        return kw_model
    except Exception as e:
        st.error(f"⚠️ KeyBERT 로딩 실패: {e}")
        return None
# ============================================================================
# 유틸리티 함수들
# ============================================================================
def parse_kakao_txt(text: str) -> List[Dict[str, Any]]:
    """카카오톡 텍스트 파싱"""
    messages = []
    
    pattern1 = r'(\d{4}년\s+\d{1,2}월\s+\d{1,2}일.*?),\s*(.+?)\s*:\s*(.+)'
    pattern2 = r'(\d{4}\.\d{1,2}\.\d{1,2}\s+\d{1,2}:\d{2}:\d{2}\s+[AP]M),\s*(.+?)\s*:\s*(.+)'
    
    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        match = re.match(pattern1, line)
        if not match:
            match = re.match(pattern2, line)
        
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

# ============================================================================
# Agent 함수들
# ============================================================================

def preprocessor_agent(state: AppState, classifier) -> AppState:
    """전처리 에이전트 - CSV 파싱 및 구조화"""
    raw_input = state.get("raw_input", "")
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
    
    state["text"] = text
    state["messages"] = messages
    state["completed_agents"] = ["preprocessor"]
    
    return state

def emotion_agent(state: AppState, classifier) -> AppState:
    """감정 분석 에이전트"""
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
    
    emotion_summary = {
        "total_msgs": total_msgs,
        "counts": dict(emotion_counts),
        "ratios": emotion_ratios,
        "dominant_label": dominant_label,
        "speaker_analysis": speaker_analysis
    }
    
    state["messages"] = messages
    state["emotion_df"] = emotion_df
    state["emotion_summary"] = emotion_summary
    state["completed_agents"] = state.get("completed_agents", []) + ["emotion"]
    
    return state

def insight_agent(state: AppState, openai_client) -> AppState:
    """인사이트 생성 에이전트 (병렬 실행 1)"""
    emotion_summary = state.get("emotion_summary", {})
    emotion_df = state.get("emotion_df", pd.DataFrame())
    speaker_analysis = emotion_summary.get("speaker_analysis", {})
    
    if not openai_client:
        state["insight_text"] = "⚠️ OpenAI API 키가 필요합니다."
        state["completed_agents"] = state.get("completed_agents", []) + ["insight"]
        return state
    
    try:
        prompt = f"""다음은 대화 참여자들의 감정 분석 결과입니다:

📊 전체 집계:
- 총 메시지: {emotion_summary.get('total_msgs', 0)}개
- 주요 감정: {emotion_summary.get('dominant_label', '중립')}
- 감정 분포: {emotion_summary.get('counts', {})}

👥 화자별 분석:
{json.dumps(speaker_analysis, ensure_ascii=False, indent=2)}

위 내용을 바탕으로 **500자 이내**로 핵심 요약을 작성해주세요.

아래 1~3번을 각각 한 세 문장씩, 총 9문장으로 작성합니다.

1. 전반적인 감정 패턴:
   - 대화의 시작–중간–끝 흐름을 기준으로, 어떤 감정이 어떻게 변화했는지 서술하세요.
   - 단순히 '기쁨이 많다'가 아니라, 예를 들어 '초반엔 피로와 걱정이 두드러지지만, 중간 이후 서로의 위로로 분위기가 점점 안정됩니다'처럼 **감정 전환**이 드러나게 써주세요.

2. 화자 간 관계 특성:
   - 두 화자의 역할 차이(예: 한쪽은 고민을 털어놓고, 다른 쪽은 위로·조언 중심인지), 공감·지지·갈등 여부를 중심으로 관계를 한 문장으로 정리하세요.
   - '사이가 좋다' 수준이 아니라, 예를 들어 'A는 솔직하게 힘든 감정을 털어놓고, B는 이를 진지하게 받아들이며 공감해주는 관계입니다'처럼 **상호작용의 특징**이 드러나게 써주세요.

3. 간단한 조언:
   - 1–2번에서 드러난 감정 흐름과 관계 특성을 바탕으로, 두 사람이 감정을 더 건강하게 나누거나 관계를 유지·개선하는 데 도움이 될 만한 **구체적인 행동 한 가지**를 제안하세요.
   - 바로 실천 가능한 현실적인 조언으로, 문장 끝에 이모지 1개를 붙여주세요.

형식 규칙:
- 각 항목은 '1. 문장', '2. 문장', '3. 문장' 형태로, 한 줄에 3문장씩만 적습니다.
- 총 9문장만 출력합니다.
- 한국어 존댓말을 사용합니다.
- 전체 분량은 500자를 넘기지 마세요.
"""
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 감정 분석 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        state["insight_text"] = response.choices[0].message.content
    
    except Exception as e:
        state["insight_text"] = f"⚠️ 인사이트 생성 오류: {e}"
    
    state["completed_agents"] = state.get("completed_agents", []) + ["insight"]
    return state

def summary_agent(state: AppState) -> AppState:
    """통계 요약 에이전트 (병렬 실행 2)"""
    emotion_df = state.get("emotion_df", pd.DataFrame())
    messages = state.get("messages", [])
    
    if emotion_df.empty:
        state["completed_agents"] = state.get("completed_agents", []) + ["summary"]
        return state
    
    # 시간대별 분석
    time_distribution = {}
    for msg in messages:
        datetime_str = msg.get("datetime", "")
        if "AM" in datetime_str or "오전" in datetime_str:
            time_period = "오전"
        elif "PM" in datetime_str or "오후" in datetime_str:
            time_period = "오후"
        else:
            time_period = "기타"
        
        time_distribution[time_period] = time_distribution.get(time_period, 0) + 1
    
    # 메시지 길이 통계
    avg_length = emotion_df["text_length"].mean()
    max_length = emotion_df["text_length"].max()
    min_length = emotion_df["text_length"].min()
    
    # 화자 참여도
    speaker_participation = emotion_df["speaker"].value_counts().to_dict()
    
    statistical_summary = {
        "time_distribution": time_distribution,
        "message_length": {
            "average": round(avg_length, 2),
            "max": max_length,
            "min": min_length
        },
        "speaker_participation": speaker_participation,
        "total_speakers": len(emotion_df["speaker"].unique())
    }
    
    state["statistical_summary"] = statistical_summary
    state["completed_agents"] = state.get("completed_agents", []) + ["summary"]
    
    return state


def keyword_extractor_agent(state: AppState) -> AppState:
    """키워드 추출 에이전트 (병렬 실행 3) - 개선 버전"""
    messages = state.get("messages", [])
    emotion_summary = state.get("emotion_summary", {})
    
    # 모든 메시지 텍스트 결합
    all_text = " ".join([msg.get("text", "") for msg in messages])
    
    # 확장된 불용어 사전
    stopwords = {
        # 대명사/지시어
        '이', '그', '저', '것', '나', '너', '우리', '저희', '자기', '여기', '거기', '저기',
        '이거', '그거', '저거', '뭐', '어디', '누구', '언제', '어떻게', '왜',
        # 조사/어미
        '은', '는', '이', '가', '을', '를', '의', '에', '에서', '로', '으로', '와', '과', 
        '도', '만', '라도', '부터', '까지', '보다', '처럼', '같이', '마저', '조차',
        # 동사/형용사 어간
        '하다', '되다', '있다', '없다', '이다', '아니다', '같다', '보다', '주다', '받다',
        # 부사/접속사
        '매우', '너무', '아주', '정말', '진짜', '완전', '엄청', '되게', '좀', '더', '덜',
        '그래서', '그러나', '그런데', '하지만', '또', '또한', '역시', '과연', 
        # 감탄사
        '네', '예', '응', '어', '음', '에', '아', '오', '우',
        # 기타 고빈도 무의미어
        '거', '것', '게', '때', '수', '등', '들', '안', '않', '못', '말', '데', '분',
        '점', '번', '채', '편', '쪽', '개', '명', '살', '원', '시', '일', '월', '년'
    }
    
    try:
        # 방법 1: KoNLPy Okt 사용 (더 정확)
        if use_konlpy:
            try:
                from konlpy.tag import Okt
                okt = Okt()
                
                # 명사만 추출 (가장 의미있는 단어)
                nouns = okt.nouns(all_text)
                
                # 2글자 이상 명사만 필터링
                nouns = [n for n in nouns if len(n) >= 2]
                
                # 불용어 제거 및 카운팅
                word_counts = Counter([n for n in nouns if n not in stopwords])
                
                # 감정 관련 키워드 추가 점수 (감정과 연결)
                emotion_keywords = {
                    '힘들다': 3, '피곤': 3, '스트레스': 3, '걱정': 3, '불안': 3,
                    '좋다': 2, '행복': 2, '기쁨': 2, '사랑': 2, '감사': 2,
                    '화': 2, '짜증': 2, '답답': 2, '슬프다': 2, '우울': 2
                }
                
                # 감정 키워드에 가중치 부여
                for keyword, bonus in emotion_keywords.items():
                    if keyword in word_counts:
                        word_counts[keyword] += bonus
                
            except ImportError:
                st.warning("⚠️ KoNLPy가 설치되지 않아 기본 방법을 사용합니다.")
                use_konlpy = False
        
        # 방법 2: 정규식 기반 (KoNLPy 없을 때)
        if not use_konlpy:
            # 한글만 추출 (2-10자 길이)
            words = re.findall(r'[가-힣]{2,10}', all_text)
            word_counts = Counter([w for w in words if w not in stopwords])
    
    except Exception as e:
        st.warning(f"키워드 추출 중 오류: {e}")
        words = re.findall(r'[가-힣]{2,10}', all_text)
        word_counts = Counter([w for w in words if w not in stopwords])
    
    # 상위 15개 키워드 (더 많이)
    top_keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    extracted_keywords = [word for word, count in top_keywords]
    
    # 감정별 상세 검색 쿼리 생성
    dominant = emotion_summary.get("dominant_label", "중립")
    dominant_lower = dominant.lower()
    
    # 추출된 키워드 상위 3개를 쿼리에 포함
    top_3_keywords = ' '.join(extracted_keywords[:3]) if extracted_keywords else ""
    
    # 감정별 맞춤 쿼리
    query_templates = {
        "슬픔": f"우울 슬픔 극복 심리 상담 치유 {top_3_keywords}",
        "우울": f"우울증 극복 마음 치유 심리 건강 {top_3_keywords}",
        "불안": f"불안 해소 스트레스 관리 마음챙김 명상 {top_3_keywords}",
        "걱정": f"걱정 해소 심리 안정 멘탈 케어 {top_3_keywords}",
        "분노": f"분노 조절 화 다스리기 감정 관리 {top_3_keywords}",
        "화": f"화 조절 갈등 해결 소통 방법 {top_3_keywords}",
        "기쁨": f"행복 유지 긍정 에너지 관계 개선 {top_3_keywords}",
        "행복": f"행복한 삶 긍정 마인드 자기 계발 {top_3_keywords}",
        "사랑": f"관계 유지 사랑 키우기 소통 방법 {top_3_keywords}",
        "놀람": f"감정 안정 심리 회복 마음 챙김 {top_3_keywords}",
        "혐오": f"부정 감정 극복 마음 정리 심리 상담 {top_3_keywords}",
        "공포": f"불안 극복 두려움 해소 심리 치료 {top_3_keywords}"
    }
    
    # 매칭되는 쿼리 찾기
    content_query = None
    for emotion_key, query in query_templates.items():
        if emotion_key in dominant_lower:
            content_query = query
            break
    
    # 매칭 안되면 기본 쿼리
    if not content_query:
        content_query = f"감정 관리 심리 건강 멘탈 케어 {top_3_keywords}"
    
    state["extracted_keywords"] = extracted_keywords
    state["content_query"] = content_query
    state["content_query_display"] = content_query_display
    state["completed_agents"] = state.get("completed_agents", []) + ["keyword"]
    
    return state

def aggregator_agent(state: AppState, openai_client) -> AppState:
    """최종 통합 에이전트 - 모든 결과를 종합 정리"""
    
    emotion_summary = state.get("emotion_summary", {})
    statistical_summary = state.get("statistical_summary", {})
    insight_text = state.get("insight_text", "")
    keywords = state.get("extracted_keywords", [])
    content_recos = state.get("content_recos", [])
    
    # 통합 리포트 생성
    final_report = {
        "overview": {
            "total_messages": emotion_summary.get("total_msgs", 0),
            "total_speakers": statistical_summary.get("total_speakers", 0),
            "dominant_emotion": emotion_summary.get("dominant_label", "중립"),
            "time_distribution": statistical_summary.get("time_distribution", {})
        },
        "emotion_analysis": emotion_summary,
        "statistics": statistical_summary,
        "keywords": keywords[:5],  # 상위 5개
        "insight": insight_text,
        "recommendations": content_recos[:3]  # 상위 3개
    }
    
    # OpenAI로 최종 요약 생성 (선택)
    if openai_client:
        try:
            prompt = f"""다음은 대화 분석의 모든 결과입니다:

📊 **기본 정보**
- 전체 메시지: {final_report['overview']['total_messages']}개
- 참여자: {final_report['overview']['total_speakers']}명
- 주요 감정: {final_report['overview']['dominant_emotion']}

🔑 **핵심 키워드**
{', '.join(keywords[:5])}

💡 **심리 분석**
{insight_text}

위 내용을 바탕으로 **300자 이내**로 핵심 요약을 작성해주세요:
1. 대화의 주요 특징 (2줄)
2. 감정적 핵심 (2줄)
특히 감정적 핵심에는 시간에 따른 상황을 반영해서 어떻게 관계나 상황이 달라졌는지, 최종적으로는 어떻게 마무리가 됐는지 자세하게 작성해줘.
3. 한 줄 조언 (2줄)

이모지와 함께 간결하게 작성해주세요."""
            
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "당신은 데이터 분석 요약 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=700,
                temperature=0.7
            )
            
            final_report["summary"] = response.choices[0].message.content
        
        except Exception as e:
            final_report["summary"] = f"⚠️ 요약 생성 오류: {e}"
    else:
        # OpenAI 없으면 간단한 요약
        final_report["summary"] = f"""
📊 **분석 요약**

총 {final_report['overview']['total_messages']}개의 메시지를 분석했습니다.
{final_report['overview']['total_speakers']}명이 대화에 참여했으며, 주요 감정은 **{final_report['overview']['dominant_emotion']}**입니다.

핵심 키워드: {', '.join(keywords[:3])}
        """
    
    state["final_report"] = final_report
    state["completed_agents"] = state.get("completed_agents", []) + ["aggregator"]
    
    return state

# ============================================================================
# 분석 실행 함수
# ============================================================================
def run_analysis(text: str, openai_key: str, tavily_key: str, classifier):
    """전체 분석 파이프라인 실행"""
    
    openai_client = init_openai_client(openai_key) if openai_key else None
    
    initial_state = {"raw_input": text}
    
    # 1. Preprocessor
    with st.status("🔄 PreprocessorAgent 실행 중...", expanded=True) as status:
        st.write("CSV 파싱 및 전처리 중...")
        state = preprocessor_agent(initial_state, classifier)
        st.write(f"✅ {len(state['messages'])}개 메시지 파싱 완료")
        status.update(label="✅ Preprocessor 완료", state="complete")
    
    # 2. Emotion Agent
    with st.status("🟢 EmotionAgent 실행 중...", expanded=True) as status:
        st.write("감정 분석 중...")
        state = emotion_agent(state, classifier)
        st.write(f"✅ 주요 감정: {state['emotion_summary']['dominant_label']}")
        status.update(label="✅ EmotionAgent 완료", state="complete")
    
    # 3. 병렬 실행 (Insight + Summary + Keyword)
    with st.status("⚡ 병렬 에이전트 실행 중...", expanded=True) as status:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("🟡 InsightAgent...")
            state = insight_agent(state, openai_client)
            st.write("✅ 완료")
        
        with col2:
            st.write("🔵 SummaryAgent...")
            state = summary_agent(state)
            st.write("✅ 완료")
        
        with col3:
            st.write("🟣 KeywordAgent...")
            state = keyword_extractor_agent(state)
            st.write("✅ 완료")
        
        status.update(label="✅ 병렬 실행 완료", state="complete")
    
    # 4. Content Agent
    if tavily_key:
        with st.status("🔴 ContentAgent 실행 중...", expanded=True) as status:
            st.write("콘텐츠 추천 중...")
            state = content_agent(state, tavily_key)
            st.write(f"✅ {len(state.get('content_recos', []))}개 추천 완료")
            status.update(label="✅ ContentAgent 완료", state="complete")
    
    # 5. Aggregator (최종 통합)
    with st.status("📊 AggregatorAgent 실행 중...", expanded=True) as status:
        st.write("최종 리포트 생성 중...")
        state = aggregator_agent(state, openai_client)
        st.write("✅ 통합 리포트 생성 완료")
        status.update(label="✅ Aggregator 완료", state="complete")
    
    return state

# ============================================================================
# Streamlit UI
# ============================================================================
def main():
    st.title("🎭 감정 분석 멀티에이전트 시스템")
    st.markdown("### 병렬 실행 아키텍처 + Aggregator")
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        openai_key = st.text_input("OpenAI API Key", type="password", 
                                   help="선택 - 인사이트 및 통합 요약 생성")
        tavily_key = st.text_input("Tavily API Key", type="password",
                                   help="선택 - 콘텐츠 추천")
        
        st.divider()
        
        # 모델 로딩
        # 모델 로딩
        if not st.session_state.models_loaded:
            if st.button("🚀 모델 로딩", use_container_width=True):
                with st.spinner("모델 로딩 중... (약 1-2분 소요)"):
                    # 감정 분석 모델
                    st.session_state.emotion_classifier = load_emotion_model()
                    # KeyBERT 모델
                    st.session_state.keybert_model = load_keybert_model()
                    
                    if st.session_state.emotion_classifier:
                        st.session_state.models_loaded = True
                        st.success("✅ 모델 로딩 완료!")
                        if st.session_state.keybert_model:
                            st.success("✅ KeyBERT 로딩 완료!")
                        else:
                            st.warning("⚠️ KeyBERT 로딩 실패")
                        st.rerun()
                    else:
                        st.error("❌ 모델 로딩 실패")
        else:
            st.success("✅ 모델 로딩됨")
            if st.session_state.keybert_model:
                st.success("✅ KeyBERT 활성화")
            if st.button("🔄 모델 재로딩", use_container_width=True):
                st.session_state.models_loaded = False
                st.session_state.emotion_classifier = None
                st.session_state.keybert_model = None
                st.rerun()
        
        st.divider()
        st.caption("🏗️ 아키텍처:")
        st.code("""
Preprocessor
    ↓
Emotion
    ↓
┌───┼───┬─────┐
I   S   K    (병렬)
└───┼───┴─────┘
    ↓
Content
    ↓
Aggregator 📊
        """)
    
    # 메인 영역
    if not st.session_state.models_loaded:
        st.warning("⚠️ 먼저 사이드바에서 모델을 로딩해주세요.")
        return
    
    st.header("📝 대화 입력")
    
    # 입력 방식 선택
    input_method = st.radio(
        "입력 방식 선택",
        ["📁 CSV 파일 업로드", "✍️ 직접 입력"],
        horizontal=True
    )
    
    input_text = ""
    
    if input_method == "📁 CSV 파일 업로드":
        st.info("💡 카카오톡 → 대화방 → 설정(≡) → '대화 내보내기' → CSV 파일 저장")
        st.caption("📋 필수 컬럼: `date`, `user`, `message`")
        
        uploaded_file = st.file_uploader(
            "카카오톡 대화 CSV 파일 업로드",
            type=['csv'],
            help="date, user, message 컬럼이 있는 CSV 파일"
        )
        
        if uploaded_file is not None:
            try:
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='cp949')
                
                st.success(f"✅ CSV 파일 로드 완료: {uploaded_file.name}")
                
                with st.expander("📊 CSV 데이터 미리보기"):
                    st.dataframe(df.head(10), use_container_width=True)
                    st.caption(f"총 {len(df)}개 행")
                
                df.columns = df.columns.str.lower().str.strip()
                
                if 'date' in df.columns and 'user' in df.columns and 'message' in df.columns:
                    lines = []
                    for _, row in df.iterrows():
                        date = str(row['date']).strip()
                        user = str(row['user']).strip()
                        message = str(row['message']).strip()
                        
                        if message and message != 'nan':
                            lines.append(f"{date}, {user} : {message}")
                    
                    input_text = '\n'.join(lines)
                    st.info(f"✅ {len(lines)}개 메시지 변환 완료")
                    
                    with st.expander("📝 변환된 텍스트 미리보기 (처음 5줄)"):
                        st.text('\n'.join(lines[:5]))
                        if len(lines) > 5:
                            st.caption(f"... 외 {len(lines) - 5}줄")
                else:
                    st.error(f"❌ CSV 파일에 'date', 'user', 'message' 컬럼이 필요합니다.\n\n현재 컬럼: {list(df.columns)}")
                    
            except Exception as e:
                st.error(f"❌ 파일 읽기 오류: {e}")
    
    else:  # 직접 입력
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
            st.error("입력 텍스트를 입력하거나 CSV 파일을 업로드해주세요.")
            return
        
        # 분석 실행
        result = run_analysis(
            input_text, 
            openai_key, 
            tavily_key,
            st.session_state.emotion_classifier
        )
        
        st.session_state.analysis_result = result
        st.success("✅ 분석 완료!")
    
    # 결과 표시
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        final_report = result.get("final_report", {})
        
        st.divider()
        
        # 📊 Aggregator가 만든 최종 요약 (제일 위에 표시)
        st.header("📊 종합 분석 리포트")
        
        summary = final_report.get("summary", "")
        if summary:
            st.markdown(summary)
        
        # 주요 지표 카드
        overview = final_report.get("overview", {})
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📝 전체 메시지", overview.get("total_messages", 0))
        with col2:
            st.metric("👥 참여자", overview.get("total_speakers", 0))
        with col3:
            st.metric("😊 주요 감정", overview.get("dominant_emotion", "중립"))
        with col4:
            time_dist = overview.get("time_distribution", {})
            main_time = max(time_dist, key=time_dist.get) if time_dist else "N/A"
            st.metric("⏰ 주요 시간대", main_time)
        
        st.divider()
        
        # 탭으로 상세 결과 구분
        tabs = st.tabs(["📈 감정 분석", "💡 인사이트", "📊 통계", "🔑 키워드", "🎬 추천 콘텐츠"])
        
        # 📈 감정 분석 탭
        with tabs[0]:
            emotion_summary = result.get("emotion_summary", {})
            emotion_df = result.get("emotion_df", pd.DataFrame())
            
            st.subheader("📊 감정 분포")
            emotion_counts = emotion_summary.get("counts", {})
            if emotion_counts:
                chart_df = pd.DataFrame(list(emotion_counts.items()), 
                                       columns=["감정", "개수"])
                st.bar_chart(chart_df.set_index("감정"))
            
            st.subheader("👥 화자별 분석")
            speaker_analysis = emotion_summary.get("speaker_analysis", {})
            for speaker, data in speaker_analysis.items():
                with st.expander(f"**{speaker}** - {data['message_count']}개 메시지"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("주요 감정", data['dominant_emotion'])
                    with col2:
                        st.metric("평균 점수", f"{data['avg_score']:.3f}")
                    st.write(f"**감정 분포:** {data['emotion_distribution']}")
            
            st.subheader("📋 메시지별 상세")
            if not emotion_df.empty:
                st.dataframe(emotion_df, use_container_width=True)
        
        # 💡 인사이트 탭
        with tabs[1]:
            insight_text = result.get("insight_text", "")
            if insight_text:
                st.markdown(insight_text)
            else:
                st.info("💡 OpenAI API 키를 입력하면 더 상세한 인사이트를 제공합니다.")
        
        # 📊 통계 탭
        with tabs[2]:
            statistical_summary = result.get("statistical_summary", {})
            
            st.subheader("⏰ 시간대별 분포")
            time_dist = statistical_summary.get("time_distribution", {})
            if time_dist:
                time_df = pd.DataFrame(list(time_dist.items()), 
                                      columns=["시간대", "메시지 수"])
                st.bar_chart(time_df.set_index("시간대"))
            
            st.subheader("📏 메시지 길이 통계")
            msg_length = statistical_summary.get("message_length", {})
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("평균 길이", f"{msg_length.get('average', 0)}자")
            with col2:
                st.metric("최대 길이", f"{msg_length.get('max', 0)}자")
            with col3:
                st.metric("최소 길이", f"{msg_length.get('min', 0)}자")
            
            st.subheader("👥 화자별 참여도")
            speaker_participation = statistical_summary.get("speaker_participation", {})
            if speaker_participation:
                part_df = pd.DataFrame(list(speaker_participation.items()), 
                                      columns=["화자", "메시지 수"])
                st.bar_chart(part_df.set_index("화자"))
        
        # 🔑 키워드 탭
        with tabs[3]:
            keywords = result.get("extracted_keywords", [])
            
            st.subheader("🔑 추출된 핵심 키워드")
            if keywords:
                # 키워드를 컬럼으로 나누어 표시 (가독성 향상)
                cols = st.columns(3)
                for idx, kw in enumerate(keywords[:10]):
                    col_idx = idx % 3
                    with cols[col_idx]:
                        st.markdown(f"""
                        <div style='
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white;
                            padding: 12px 20px;
                            margin: 8px 0;
                            border-radius: 20px;
                            text-align: center;
                            font-weight: bold;
                            font-size: 16px;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                        '>
                            #{kw}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("키워드가 추출되지 않았습니다.")
            
            st.divider()
            
            content_query = result.get("content_query", "")
            if content_query:
                st.subheader("🔍 콘텐츠 검색 쿼리")
                st.info(f"**{content_query}**")
                st.caption("👆 이 키워드로 관련 콘텐츠를 추천합니다")
        
        # 🎬 추천 콘텐츠 탭
        with tabs[4]:
            content_recos = result.get("content_recos", [])
            
            if content_recos:
                st.subheader(f"🎬 추천 콘텐츠 ({len(content_recos)}개)")
                
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
                st.warning("💡 Tavily API 키를 입력하면 관련 콘텐츠를 추천해드립니다.")

if __name__ == "__main__":
    main()