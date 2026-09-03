# 🎬 Korean Movie Recommendation System (가상 한국영화 3종 추천 모델 비교)

가상 한국영화 500편의 비정형 텍스트(줄거리, 장르, 키워드)를 전처리하고, 자연어 처리(NLP) 3대 기법인 **TF-IDF**, **Word2Vec**, **Sentence-BERT** 모델을 적용하여 영화 추천 결과를 나란히 비교하는 Streamlit 웹 애플리케이션입니다.

---

## 🚀 주요 기능
- **영화 검색 & 선택**: 500편의 가상 한국영화 중 원하는 작품 선택/검색
- **3종 추천 모델 병렬 비교**:
  1. 📊 **TF-IDF (unigram + bigram)**: 형태소 기반 단어 빈도 및 키워드 일치 기반 추천
  2. 🌐 **Word2Vec (Mean Pooling)**: 형태소 임베딩 학습 + 단어 벡터 평균 기반 토픽 추천
  3. 🤖 **Sentence-BERT (`jhgan/ko-sroberta-multitask`)**: 문맥 및 시맨틱 뉘앙스 기반 심층 추천
- **추천 결과 상세**: 모델별 Top 5 추천 영화, 장르, 유사도 점수 프로그레스 바, 줄거리 요약 카드 제공

---

## 📦 데이터셋 구성
- `clean_movies.csv`: 500편 영화의 정제 줄거리(`overview`), 장르(`genres_str`), 키워드(`keywords_str`) 및 결합 텍스트(`model_text`)
- 원본 엑셀 및 비공개 데이터는 포함하지 않는 독립 경량 데이터셋

---

## 🛠️ 로컬 실행 방법
```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. Streamlit 웹앱 실행
streamlit run app.py
```
