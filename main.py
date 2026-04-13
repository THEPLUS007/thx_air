from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
import re
from typing import Any, Dict, List, Optional
import asyncio
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None
    create_client = None
from sentence_transformers import SentenceTransformer

# 환경 변수 로드
load_dotenv()

# Gemma 4 클라이언트 설정
# .env 파일에 GOOGLE_API_KEY가 저장되어 있어야 합니다.
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Supabase 클라이언트 설정
if SUPABASE_AVAILABLE:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("경고: SUPABASE_URL 및 SUPABASE_KEY가 설정되지 않았습니다. DB 기능이 비활성화됩니다.")
        supabase = None
    else:
        try:
            supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception as e:
            print(f"경고: Supabase 연결 실패: {e}. DB 기능이 비활성화됩니다.")
            supabase = None
else:
    print("경고: Supabase 라이브러리를 불러올 수 없습니다. DB 기능이 비활성화됩니다.")
    supabase = None

# Sentence Transformer for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Gemma 4 모델 글로벌 초기화
gemma_model = genai.GenerativeModel("gemma-4-31b-it")

app = FastAPI(
    title="AI 여행 플래너",
    description="Gemma 4 모델을 사용한 여행 계획 생성 백엔드",
    timeout=300
)

templates = Jinja2Templates(directory="templates")

class TripQuery(BaseModel):
    query: str
    budget: Optional[str] = None
    people: Optional[int] = None

class CreatePlanRequest(BaseModel):
    user_id: str
    query: str

class UpdatePlanRequest(BaseModel):
    plan_id: str
    user_id: str
    query: str


def get_response_text(response: Any) -> str:
    if hasattr(response, 'parts') and response.parts:
        if len(response.parts) > 1 and getattr(response.parts[1], 'text', None):
            return response.parts[1].text.strip()
        return getattr(response.parts[0], 'text', '').strip()
    if hasattr(response, 'text'):
        return response.text.strip()
    raise Exception('모델 응답에서 텍스트를 찾을 수 없습니다')


def extract_json_from_text(result_text: str) -> dict:
    if '```json' in result_text:
        result_text = result_text.split('```json', 1)[1].split('```', 1)[0].strip()
    elif '```' in result_text:
        result_text = result_text.split('```', 1)[1].split('```', 1)[0].strip()

    match = re.search(r'\{.*\}', result_text, re.DOTALL)
    if not match:
        raise ValueError('JSON 객체를 응답에서 찾을 수 없습니다')

    json_text = match.group(0)
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        cleaned = json_text.replace("'", '"')
        cleaned = re.sub(r',\s*}', '}', cleaned)
        cleaned = re.sub(r',\s*\]', ']', cleaned)
        return json.loads(cleaned)


def extract_code_block(text: str, block_type: str = None) -> str:
    """마크다운 코드 블록 추출 (Mermaid, Python 등)"""
    if not text:
        return ""
    
    text = text.strip()
    
    if block_type and f'```{block_type}' in text:
        match = re.search(rf'```{block_type}\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    if '```' in text:
        match = re.search(r'```\s*([a-z]*)\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            return match.group(2).strip()
    
    return text


def detect_language(query: str) -> str:
    """쿼리에서 언어 감지: 한국어 문자가 있으면 'ko', 아니면 'en'"""
    if re.search(r'[가-힣]', query):
        return 'ko'
    return 'en'


def generate_embedding(text: str) -> List[float]:
    return embedding_model.encode(text).tolist()


def process_pipeline(query: str) -> Dict[str, Any]:
    raw_data = planner(query)
    normalized_data = normalize_data(raw_data)
    language = detect_language(query)
    text_for_embedding = f"{normalized_data.get('destination', '')} {normalized_data.get('preferences', '')} {json.dumps(normalized_data, ensure_ascii=False)}"
    embedding = generate_embedding(text_for_embedding)
    return {
        'normalized_data': normalized_data,
        'embedding': embedding,
        'text_for_embedding': text_for_embedding,
        'language': language
    }


def planner(query: str) -> dict:
    prompt = f"""당신은 한국 사용자의 여행 계획을 파악하는 AI입니다. 다음 사용자의 말을 분석하여 꼭 필요한 정보를 JSON으로만 추출하세요.

사용자의 말: '{query}'

반드시 다음 JSON 형식으로만 답변하세요. 다른 설명은 절대 하지 말기:
{{"destination": "여행지명", "duration": "기간", "preferences": "취향", "activities": ["활동1","활동2","활동3"], "tags": ["태그1","태그2"]}}

추출 규칙:
- destination: 구체적인 도시/지역명 또는 두 도시 모두 포함 (예: 일본 (오사카 & 도쿄))
- duration: 숫자와 단위 (예: 3박4일, 5일)
- preferences: 여행 스타일 및 상황 (예: 힐링, 부모님 동반, 가성비)
- activities: 사용자가 요청한 추천 활동들
- tags: 부모님여행, 온천, 미식, 힐링, 가성비 등 쿼리에서 유추되는 대표 태그
- 예산 언급이 있으면 tags에 포함하거나 preferences에 반영
- 가능한 한 구체적으로 추론하여 채우기
- 응답은 오직 JSON 객체 하나뿐이어야 함"""
    try:
        response = gemma_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=500,
            ),
            request_options={"timeout": 120}
        )
        result_text = get_response_text(response)
        if not result_text:
            raise Exception("응답 텍스트가 비어있습니다")
        return extract_json_from_text(result_text)
    except Exception as e:
        raise Exception(f"Planner error: {str(e)}")


def structurer(data: Dict[str, Any], language: str = 'en') -> str:
    if language == 'ko':
        prompt = f"""당신은 한국 사용자의 여행 계획을 작성하는 AI입니다. 다음 여행 정보를 바탕으로 한국어로 상세한 여행 가이드를 작성하세요.

여행 정보:
- 목적지: {data.get('destination', 'N/A')}
- 기간: {data.get('duration', 'N/A')}
- 취향: {data.get('preferences', 'N/A')}
- 추천 활동: {', '.join(data.get('activities', []))}
- 태그: {', '.join(data.get('tags', []))}
- 예산: {data.get('budget', 'N/A')}
- 인원: {data.get('people', '1')}

한국어로 상세한 여행 가이드를 작성하세요. YAML frontmatter를 포함하고, 마크다운 형식으로 작성하세요.

---
destination: {data.get('destination', 'N/A')}
duration: {data.get('duration', 'N/A')}
preferences: {data.get('preferences', 'N/A')}
budget: {data.get('budget', 'N/A')}
people: {data.get('people', '1')}
tags: [{', '.join(data.get('tags', []))}]
---

## 목적지 소개
## 추천 일정
## 숙박 추천
## 예산 안내
## 팁
"""
    else:
        prompt = f"""You are an AI that writes detailed travel guides in English. Use the travel information below to create a complete Obsidian-style Markdown guide.

Travel information:
- Destination: {data.get('destination', 'N/A')}
- Duration: {data.get('duration', 'N/A')}
- Preferences: {data.get('preferences', 'N/A')}
- Recommended activities: {', '.join(data.get('activities', []))}
- Tags: {', '.join(data.get('tags', []))}
- Budget: {data.get('budget', 'N/A')}
- People: {data.get('people', '1')}

Create English markdown with the following requirements:
- Include YAML frontmatter at the top
- Use headings (##) for sections
- Include destination introduction, accommodation recommendations, healing content, and travel tips
- Mention budget or parent-friendly details when relevant
- Do not include any Korean

Example frontmatter:
---
destination: {data.get('destination', 'N/A')}
duration: {data.get('duration', 'N/A')}
preferences: {data.get('preferences', 'N/A')}
budget: {data.get('budget', 'N/A')}
people: {data.get('people', '1')}
tags: [{', '.join(data.get('tags', []))}]
---

Write the guide in English only."""
    
    try:
        for attempt in range(2):
            response = gemma_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=2000,
                ),
                request_options={"timeout": 120}
            )
            result_text = get_response_text(response)
            if result_text:
                return result_text
        raise Exception("Structurer 응답이 비어있습니다")
    except Exception as e:
        raise Exception(f"Structurer error: {str(e)}")


async def visualizer(full_content: str, json_data: Dict[str, Any]) -> str:
    """Gemma 4 기반 AI 에이전트: 여행 계획을 분석하여 Mermaid 시각화 생성"""
    
    destination = json_data.get('destination', '목적지')
    duration = json_data.get('duration', '기간 미상')
    people = json_data.get('people', '1')
    
    # 콘텐츠에서 일정 섹션만 추출 (효율성)
    schedule_section = full_content
    if '## 추천 일정' in full_content or '## 일정' in full_content:
        match = re.search(r'(##\s*추천 일정|##\s*일정).*?(?=##|\Z)', full_content, re.DOTALL)
        if match:
            schedule_section = match.group(0)
    
    prompt = f"""당신은 여행 계획을 시각적으로 표현하는 AI 에이전트입니다.

### 기본 정보
- 목적지: {destination}
- 기간: {duration}
- 인원: {people}명

### 여행 계획 일정
{schedule_section[:1800]}{'(내용 축약)' if len(schedule_section) > 1800 else ''}

### 요구사항
위 여행 계획을 기반으로 Mermaid graph TD로 실제 여행 일정을 표현하세요.

**구성 규칙:**
1. 날짜별 subgraph (예: subgraph Day1["📅 Day 1: 도착"])
2. 노드: 구체적 활동명 + 이모지 (예: "✈️ 공항 도착", "🍽️ 아침 식사", "💤 호텔 휴식")
3. 이모지: ✈️비행, 🚗차량, 🚕택시 / 🍽️식사, 💤수면, 💆마사지, 🏛️관광, 🛍️쇼핑, 📸사진
4. 간단명료한 연결선

**중요:** 순수 Mermaid 코드만 출력. 마크다운 포맷팅 없음.

예시:
graph TD
    subgraph Day1["📅 Day 1"]
        A["✈️ 공항 도착"]
        B["🚕 호텔 이동"]
    end
    A --> B"""
    
    try:
        response = await asyncio.to_thread(
            gemma_model.generate_content,
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=1500,
            ),
            request_options={"timeout": 120}
        )
        
        result_text = get_response_text(response)
        if not result_text:
            raise Exception("Visualizer 응답이 비어있습니다")
        
        mermaid_code = extract_code_block(result_text, 'mermaid')
        if not mermaid_code:
            mermaid_code = extract_code_block(result_text)
        
        if not mermaid_code.strip().startswith('graph'):
            mermaid_code = f"""graph TD
    Start["🛬 {destination} 여행"]
    Activity["📅 {duration}"]
    End["🏠 귀가"]
    Start --> Activity --> End"""
        
        return mermaid_code.strip()
    except Exception as e:
        raise Exception(f"Visualizer error: {str(e)}")


@app.post("/create_plan")
async def create_plan(request: CreatePlanRequest):
    try:
        pipeline_result = await asyncio.to_thread(process_pipeline, request.query)
        data_to_insert = {
            'user_id': request.user_id,
            'destination': pipeline_result['normalized_data'].get('destination', ''),
            'metadata': {
                **pipeline_result['normalized_data'],
                'lang_code': pipeline_result['language'],
                'content': None  # 나중에 생성된 콘텐츠 저장 가능
            },
            'embedding': pipeline_result['embedding']
        }
        response = supabase.table('travel_plans').insert(data_to_insert).execute()
        return {
            "message": "여행 계획이 성공적으로 생성되고 저장되었습니다.",
            "plan_id": response.data[0]['id'] if response.data else None,
            "data": pipeline_result['normalized_data']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파이프라인 처리 오류: {str(e)}")


@app.put("/update_plan")
async def update_plan(request: UpdatePlanRequest):
    try:
        existing = supabase.table('travel_plans').select('*').eq('id', request.plan_id).eq('user_id', request.user_id).execute()
        if not existing.data:
            raise HTTPException(status_code=404, detail="계획을 찾을 수 없습니다.")

        existing_data = existing.data[0].get('metadata', {}) or {}
        pipeline_result = await asyncio.to_thread(process_pipeline, request.query)
        merged_data = {**existing_data, **pipeline_result['normalized_data']}
        merged_data['lang_code'] = pipeline_result['language']

        text_for_embedding = json.dumps(merged_data, ensure_ascii=False)
        embedding = await asyncio.to_thread(generate_embedding, text_for_embedding)

        data_to_update = {
            'metadata': merged_data,
            'embedding': embedding
        }
        supabase.table('travel_plans').update(data_to_update).eq('id', request.plan_id).execute()

        return {
            "message": "여행 계획이 성공적으로 업데이트되었습니다.",
            "data": merged_data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"업데이트 오류: {str(e)}")


@app.post("/plan_trip")
async def plan_trip(trip_query: TripQuery):
    if not trip_query.query or not trip_query.query.strip():
        raise HTTPException(status_code=400, detail="query 값은 비어 있을 수 없습니다")
    if trip_query.people is not None and trip_query.people <= 0:
        raise HTTPException(status_code=400, detail="people 값은 1 이상이어야 합니다")

    try:
        # 1. Planner 호출
        json_data = await asyncio.to_thread(planner, trip_query.query)
        if trip_query.budget:
            json_data['budget'] = trip_query.budget
        if trip_query.people is not None:
            json_data['people'] = trip_query.people

        # 2. 언어 감지
        language = detect_language(trip_query.query)

        # 3. Structurer 호출 (먼저 완료)
        content_task = await asyncio.to_thread(structurer, json_data, language)

        # 4. Visualizer 호출 (structurer 결과 + json_data 활용)
        mermaid_task = await visualizer(content_task, json_data)

        # 5. 결과 조합
        final_content = f"{content_task}\n\n## 이동 경로\n\n```mermaid\n{mermaid_task}\n```\n"

        # 6. 파일 저장 (충돌 방지 위해 임시 파일 사용)
        import uuid
        temp_filename = f"output_travel_{uuid.uuid4().hex}.md"
        def write_file():
            with open(temp_filename, "w", encoding="utf-8") as f:
                f.write(final_content)
            import shutil
            shutil.move(temp_filename, "output_travel.md")
        await asyncio.to_thread(write_file)

        return {
            "message": "Gemma 4가 여행 계획을 성공적으로 생성했습니다.",
            "content": final_content,
            "language": language,
            "mermaid": mermaid_task
        }
    except Exception as e:
        error_msg = str(e)
        if "504" in error_msg or "Deadline" in error_msg:
            error_msg = "AI 모델 응답 시간이 초과되었습니다. 잠시 후 다시 시도해주세요."
        elif "API" in error_msg:
            error_msg = "Google API 연결에 문제가 있습니다. API 키를 확인해주세요."
        else:
            error_msg = f"처리 중 오류가 발생했습니다: {error_msg}"
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/", response_class=HTMLResponse)
async def get_web_interface(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/get_output")
async def get_output():
    try:
        with open("output_travel.md", "r", encoding="utf-8") as f:
            content = f.read()
        return {"content": content}
    except FileNotFoundError:
        return {"content": "여행 계획이 아직 생성되지 않았습니다."}
    except Exception as e:
        return {"content": f"파일 읽기 오류: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
