# thx_air

## AI 여행 플래너 백엔드 with Supabase RAG 시스템

이 프로젝트는 FastAPI를 기반으로 한 Python 백엔드로, Google Gemma 4 모델을 사용하여 AI 여행 플래너를 구현합니다. Supabase(PostgreSQL)를 메인 DB로 사용하며, RAG(Retrieval-Augmented Generation) 시스템을 구축하여 여행 계획 생성 및 수정을 지원합니다.

### 주요 기능

- **3단계 데이터 처리 파이프라인**:
  - **Extraction**: 사용자의 자연어 쿼리에서 여행 의도를 분석하여 JSON 객체 생성
  - **Alignment/Normalizing**: 생성된 JSON을 DB 저장 전 의미론적 보정 (cost → estimated_budget, location → visit_point 등)
  - **Supabase Integration**: 보정된 데이터를 travel_plans 테이블에 저장, RAG를 위한 텍스트 임베딩 생성

- **하이브리드 스키마**:
  - Fixed Fields: id, user_id, destination, created_at, updated_at
  - Flexible Fields: metadata (JSONB) - Gemma가 생성한 다채로운 데이터 저장

- **RAG 기반 업데이트**: 기존 여행 계획을 참조하여 일관성을 유지하며 수정

### 설치 및 설정

1. 의존성 설치:
   ```bash
   pip install -r requirements.txt
   ```

2. 환경 변수 설정:
   `.env` 파일에 다음 변수를 설정하세요:
   ```
   GOOGLE_API_KEY=your_google_api_key
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_KEY=your_supabase_anon_key
   ```

3. Supabase 테이블 생성:
   Supabase 대시보드에서 SQL 에디터에 `create_table.sql`의 내용을 실행하세요.

4. 서버 실행:
   ```bash
   python main.py
   ```

### API 엔드포인트

- **POST /create_plan**: 새로운 여행 계획 생성
  ```json
  {
    "user_id": "user-uuid",
    "query": "일본 오사카로 3박4일 여행 계획해줘"
  }
  ```

- **PUT /update_plan**: 기존 계획 수정 (RAG 참조)
  ```json
  {
    "plan_id": "plan-uuid",
    "user_id": "user-uuid",
    "query": "예산을 150만원으로 늘려서 계획 수정해줘"
  }
  ```

- **GET /**: 웹 인터페이스
- **POST /plan_trip**: 기존 여행 계획 생성 (호환성 유지)
- **GET /get_output**: 생성된 마크다운 출력 조회

### 데이터 구조

```sql
CREATE TABLE travel_plans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    destination TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB,
    embedding VECTOR(384)
);
```

### CLI 사용

```bash
python cli.py
```

결과는 `output_travel.md` 파일에 저장됩니다.