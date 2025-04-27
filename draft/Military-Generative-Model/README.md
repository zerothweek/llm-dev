## 개요
Military-Generative-Model은 병역법 데이터를 활용한 RAG(Retrieval-Augmented Generation) 및 Fine-Tuning 기반의 AI 모델입니다. Exaone 3.5를 베이스로 하며, 병역법 md(VectorDB) 데이터를 검색하고, 사전 학습된 Q-A 데이터셋을 활용하여 세부적인 질의응답이 가능한 모델입니다.

## 기능
- Fine-Tuning된 Q-A 모델

    - 병역법 관련 질문과 답변 데이터셋을 학습하여 높은 정확도의 답변 제공

    - 특정 상황에 맞는 맞춤형 답변 가능

- RAG 기반 병역법 데이터 검색

    - 병역법 md(VectorDB) 데이터를 기반으로 문서 검색 및 질의응답 수행

    - 벡터 임베딩을 통해 관련 법령을 효과적으로 찾고 활용

- 하이브리드 검색 방식

    - 키워드 검색과 벡터 검색을 병행하여 최적의 응답 제공

## 프로젝트 구조

Military-Generative-Model/  
│── datasets/                # 병역법 원문 및 학습 데이터  
│   ├── military_law.md      # 원문 병역법 데이터  
│── models/                  # 사전 학습된 모델  
│── langchain-streamlit/     # Streamlit UI  
│   ├── vectordb             # VectorDB  
│── thisandthat/             # 벡터 데이터베이스 구축 및 인퍼런스 체크  
│── training/                # Fine-Tuning 학습  

## 추론 방법

1. RAG 기반 VectorDB 구축

    병역법 md 데이터를 벡터화하여 검색 가능하도록 구성

    ./langchain-streamlit/ python -m streamlit run Home.py -> RAG 버전

2. Q-A 데이터 Fine-Tuning

    사전 구축된 질문-답변(Q-A) 데이터셋을 활용하여 모델 학습

    ./langchain-streamlit/ python -m streamlit run Home.py -> Tuning 버전

