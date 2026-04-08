# thx_air

## AI 여행 플래너 백엔드

이 프로젝트는 FastAPI를 기반으로 한 Python 백엔드로, Google Gemini 모델을 사용하여 AI 여행 플래너를 구현합니다. 3단계 파이프라인을 통해 사용자의 여행 계획을 생성합니다.

### 기능

- **Planner**: 사용자의 질문에서 여행지, 기간, 취향을 JSON으로 추출
- **Structurer**: JSON 데이터를 YAML 프론트매터로 변환
- **Visualizer**: YAML 데이터를 기반으로 Mermaid 다이어그램 생성

### 설치 및 실행

1. 의존성 설치:
   ```bash
   pip install -r requirements.txt
   ```

2. 환경 변수 설정:
   `.env` 파일에 `GOOGLE_API_KEY`를 설정하세요.

3. 서버 실행:
   ```bash
   python main.py
   ```

4. API 테스트:
   ```bash
   curl -X POST "http://localhost:8000/plan_trip" -H "Content-Type: application/json" -d '{"query": "일본으로 5일 여행 가고 싶어, 음식과 문화 체험을 좋아해"}'
   ```

결과는 `output_travel.md` 파일에 저장됩니다.

#"일본 오사카와 도쿄를 3박4일로 갈 것 같음  2명이 가고 부모님과 갈 예정임,  #그래서 익사이팅보단 힐링으로 부탁함 여비는 인당 100만원으로 잡았어 숙소와 #컨텐츠거리를 추천해줘 만약 컨텐츠가 유료라면 가격도 말해줘\n