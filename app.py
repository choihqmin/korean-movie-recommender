import os
import json
import re
import numpy as np
import pandas as pd
import streamlit as st

# 페이지 기본 설정
st.set_page_config(
    page_title="영화 추천시스템 (과제 3)",
    page_icon="🎬",
    layout="wide"
)

# Chrome 및 외부 번역기로 인한 React DOM 충돌 방지 메타태그 및 속성 설정
st.markdown(
    """
    <meta name="google" content="notranslate">
    <style>
        .stApp {
            translate: no !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# 데이터 및 모델 캐싱 로드 함수
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    clean_csv_path = os.path.join(current_dir, 'clean_movies.csv')
    
    if os.path.exists(clean_csv_path):
        df = pd.read_csv(clean_csv_path)
    else:
        excel_path = os.path.join(current_dir, '과제3-데이터(영화_데이터셋).xlsx')
        df = pd.read_excel(excel_path, sheet_name='가상한국영화')
        
        def parse_json_names(json_str):
            try:
                data = json.loads(json_str)
                return [item['name'] for item in data if 'name' in item]
            except Exception:
                return []
                
        df['genre_list'] = df['genres'].apply(parse_json_names)
        df['keyword_list'] = df['keywords'].apply(parse_json_names)
        df['genres_str'] = df['genre_list'].apply(lambda x: ' '.join(x))
        df['keywords_str'] = df['keyword_list'].apply(lambda x: ' '.join(x))
        
        def clean_text(text):
            if not isinstance(text, str):
                return ""
            text = re.sub(r'[\r\n\t]+', ' ', text)
            text = re.sub(r'[^가-힣a-zA-Z0-9\s.,?!]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
            
        df['clean_overview'] = df['overview'].apply(clean_text)
        df['model_text'] = df['clean_overview'] + ' ' + df['genres_str'] + ' ' + df['keywords_str']
    return df

# 실제 사전 계산된 유사도 행렬 로드 함수 (B안)
@st.cache_resource
def load_similarity_matrices():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    npz_path = os.path.join(current_dir, 'similarity_matrices.npz')
    
    if not os.path.exists(npz_path):
        st.error("⚠️ 유사도 행렬 캐시 파일(`similarity_matrices.npz`)이 존재하지 않습니다.\n\n"
                 "노트북(`과제3_영화_텍스트분석_추천시스템.ipynb`)을 먼저 실행하여 실제 모델 유사도 행렬을 생성해 주세요.")
        st.stop()
        
    data = np.load(npz_path)
    return data['tfidf_sim'], data['w2v_sim'], data['sbert_sim']

# 추천 목록 산출 헬퍼 함수
def get_top5_recommendations(title, sim_matrix, df):
    title_to_idx = {t: idx for idx, t in enumerate(df['title'])}
    if title not in title_to_idx:
        return []
    idx = title_to_idx[title]
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:5]
    
    results = []
    for rank, (movie_idx, score) in enumerate(sim_scores, 1):
        row = df.iloc[movie_idx]
        results.append({
            'rank': rank,
            'id': row['id'],
            'title': row['title'],
            'genres': row['genres_str'] if 'genres_str' in row else '',
            'score': float(score),
            'overview': str(row['overview'])
        })
    return results

# 메인 헤더
st.title("🎬 가상 한국영화 3종 추천 모델 비교 서비스")
st.caption("빅데이터분석 과제 3 | TF-IDF vs Word2Vec vs Sentence-BERT 기반 영화 추천 결과 비교")

# 데이터 및 모델 로드
df = load_data()
with st.spinner("3개 추천 모델(TF-IDF, Word2Vec, S-BERT) 유사도 행렬을 로드하는 중입니다..."):
    tfidf_sim, w2v_sim, sbert_sim = load_similarity_matrices()

st.divider()

# 영화 선택 인터페이스
col_input, col_meta = st.columns([1, 2])

with col_input:
    movie_list = df['title'].tolist()
    default_idx = 0
    selected_movie = st.selectbox(
        "🔎 좋아하는 영화를 선택하거나 검색하세요:",
        options=movie_list,
        index=default_idx
    )

selected_row = df[df['title'] == selected_movie].iloc[0]

with col_meta:
    st.markdown(f"### 📌 선택된 영화: **{selected_movie}**")
    st.markdown(f"- **장르:** `{selected_row['genres_str']}`")
    st.markdown(f"- **핵심 키워드:** `{selected_row['keywords_str']}`")
    st.markdown(f"- **줄거리:** {selected_row['overview']}")

st.divider()

# 3종 모델 추천 결과 산출
rec_tfidf = get_top5_recommendations(selected_movie, tfidf_sim, df)
rec_w2v = get_top5_recommendations(selected_movie, w2v_sim, df)
rec_sbert = get_top5_recommendations(selected_movie, sbert_sim, df)

# 3단 컬럼 레이아웃으로 결과 비교 출력
col_tf, col_w2v, col_bert = st.columns(3)

# 1. TF-IDF
with col_tf:
    st.markdown("### 📊 1. TF-IDF 추천 (Top 5)")
    st.caption("형태소 n-gram(1,2) 기반 단어 빈도 역문서 가중치")
    for item in rec_tfidf:
        with st.container(border=True):
            st.markdown(f"**{item['rank']}위. {item['title']}**")
            st.caption(f"장르: {item['genres']}")
            st.progress(min(max(item['score'], 0.0), 1.0), text=f"유사도: {item['score']:.4f}")
            st.write(item['overview'][:70] + "...")

# 2. Word2Vec
with col_w2v:
    st.markdown("### 🌐 2. Word2Vec 추천 (Top 5)")
    st.caption("형태소 임베딩 학습 + 문서 단어 벡터 평균 풀링")
    for item in rec_w2v:
        with st.container(border=True):
            st.markdown(f"**{item['rank']}위. {item['title']}**")
            st.caption(f"장르: {item['genres']}")
            st.progress(min(max(item['score'], 0.0), 1.0), text=f"유사도: {item['score']:.4f}")
            st.write(item['overview'][:70] + "...")

# 3. Sentence-BERT
with col_bert:
    st.markdown("### 🤖 3. Sentence-BERT 추천 (Top 5)")
    st.caption("한국어 사전학습 모델 기반 문맥적 시맨틱 밀집 임베딩")
    for item in rec_sbert:
        with st.container(border=True):
            st.markdown(f"**{item['rank']}위. {item['title']}**")
            st.caption(f"장르: {item['genres']}")
            st.progress(min(max(item['score'], 0.0), 1.0), text=f"유사도: {item['score']:.4f}")
            st.write(item['overview'][:70] + "...")

st.divider()

# 하단 모델별 추천 특성 요약
with st.expander("ℹ️ 3개 추천 모델의 특성 및 차이점 안내"):
    st.markdown("""
    | 모델 | 핵심 알고리즘 | 추천 특성 | 장단점 |
    | :--- | :--- | :--- | :--- |
    | **TF-IDF** | 단어 빈도 및 n-gram 희소 행렬 (Sparse Matrix) | 동일한 형태소/키워드(장르, 지명, 핵심단어)가 직접 겹치는 영화 우선 추천 | 어휘 일치에 정확하나, 유의어나 문맥 파악에는 한계 |
    | **Word2Vec** | 단어 분산 표상 평균 (Mean Pooling) | 단어들의 잠재 의미 공간 상 거리를 반영하여 전반적인 토픽 유사도 포착 | 단어 순서나 복합 문맥을 완전히 반영하지 못하는 압축 손실 존재 |
    | **Sentence-BERT** | Transformer 양방향 어텐션 기반 문맥 임베딩 (Dense Vector) | 줄거리 문장의 전체적인 스토리 흐름과 뉘앙스, 시맨틱 문맥을 가장 정교하게 포착 | 문맥적 연관성이 뛰어나나 모델 크기 및 연산 비용이 상대적으로 큼 |
    """)

# 제작자 정보 footer
st.markdown(
    """
    <div style="text-align: center; color: #888888; font-size: 0.85rem; margin-top: 30px; margin-bottom: 10px;">
        제주대학교 경영정보학과 최하민
    </div>
    """,
    unsafe_allow_html=True
)
