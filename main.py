from fastapi.responses import StreamingResponse
import io
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
import re
from typing import Any, Dict, List, Optional
import asyncio
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer

# 환경 변수 로드
load_dotenv()

# Gemma 4 클라이언트 설정
# .env 파일에 GOOGLE_API_KEY가 저장되어 있어야 합니다.
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Supabase 클라이언트 설정
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL 및 SUPABASE_KEY가 .env에 설정되어 있어야 합니다.")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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


def visualizer(data: Dict[str, Any]) -> str:
    destination = data.get('destination', '목적지')
    activities = data.get('activities', [])

    def split_destination(dest_text: str) -> List[str]:
        if '(' in dest_text and ')' in dest_text:
            inner = re.search(r'\((.*?)\)', dest_text)
            if inner:
                city_candidates = re.split(r'\s*(?:,|&|and|와|과)\s*', inner.group(1))
            else:
                city_candidates = re.split(r'\s*(?:,|&|and|와|과)\s*', dest_text)
        else:
            city_candidates = re.split(r'\s*(?:,|&|and|와|과)\s*', dest_text)
        cities = [re.sub(r'^[^\w가-힣]+|[^\w가-힣]+$', '', city).strip() for city in city_candidates]
        return [city for city in cities if city]

    cities = split_destination(destination)
    nodes: List[str] = ['Start["🛬 여행 시작"]']
    edges: List[str] = []

    if len(cities) > 1:
        city_nodes = []
        for idx, city in enumerate(cities):
            node_id = f'City{idx}'
            label = city if len(city) <= 18 else city[:15] + '...'
            nodes.append(f'{node_id}["🏙️ {label} 일정"]')
            city_nodes.append(node_id)
        nodes.append('End["🏠 귀가"]')
        edges.append('Start --> City0')
        for idx in range(1, len(city_nodes)):
            edges.append(f'{city_nodes[idx-1]} --> {city_nodes[idx]}')
        edges.append(f'{city_nodes[-1]} --> End')
    else:
        nodes.append(f'Arrive["✈️ {destination} 도착"]')
        edges.append('Start --> Arrive')
        for i, activity in enumerate(activities[:5], 1):
            node_id = f'Activity{i}'
            activity_short = activity[:15] + '...' if len(activity) > 15 else activity
            nodes.append(f'{node_id}["{i}. {activity_short}"]')
            if i == 1:
                edges.append(f'Arrive --> {node_id}')
            else:
                edges.append(f'Activity{i-1} --> {node_id}')
        nodes.append('End["🏠 귀가"]')
        if activities:
            edges.append(f'Activity{min(len(activities), 5)} --> End')
        else:
            edges.append('Arrive --> End')

    mermaid_code = 'graph TD\n    ' + '\n    '.join(nodes) + '\n    ' + '\n    '.join(edges)
    return mermaid_code


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

        # 3. 병렬 처리: structurer와 visualizer 동시에 실행
        async def run_structurer():
            try:
                return await asyncio.to_thread(structurer, json_data, language)
            except Exception as e:
                return f"Error in structurer: {str(e)}"

        async def run_visualizer():
            try:
                return await asyncio.to_thread(visualizer, json_data)
            except Exception as e:
                return f"Error in visualizer: {str(e)}"

        content_task, mermaid_task = await asyncio.gather(run_structurer(), run_visualizer())

        # 4. 결과 조합
        final_content = f"{content_task}\n\n## 이동 경로\n\n```mermaid\n{mermaid_task}\n```\n"

        # 5. 파일 저장 (충돌 방지 위해 임시 파일 사용)
        import uuid
        temp_filename = f"output_travel_{uuid.uuid4().hex}.md"
        def write_file():
            with open(temp_filename, "w", encoding="utf-8") as f:
                f.write(final_content)
            # 기존 파일 교체
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
