from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai  # 최신 SDK 사용
import os
from dotenv import load_dotenv
import yaml
import json

# 환경 변수 로드
load_dotenv()

# Gemma 4 클라이언트 설정
# .env 파일에 GOOGLE_API_KEY가 저장되어 있어야 합니다.
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI(title="AI 여행 플래너", description="Gemma 4 모델을 사용한 여행 계획 생성 백엔드")

class TripQuery(BaseModel):
    query: str

def planner(query: str) -> dict:
    """사용자 질문에서 핵심 여행 데이터를 JSON으로 추출 (Gemma 4 활용)"""
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
        model = genai.GenerativeModel("gemma-4-31b-it")
        response = model.generate_content(prompt)
        
        result_text = None
        if hasattr(response, 'parts') and response.parts:
            # Gemini 응답은 첫 번째 파트에 프롬프트 요약, 두 번째 파트에 실제 출력이 들어감
            if len(response.parts) > 1 and response.parts[1].text:
                result_text = response.parts[1].text.strip()
            else:
                result_text = response.parts[0].text.strip()
        elif hasattr(response, 'text'):
            result_text = response.text.strip()
        
        if not result_text:
            raise Exception("응답 텍스트가 비어있습니다")
        
        # JSON 코드 블록 제거 및 파싱
        if '```json' in result_text:
            result_text = result_text.split('```json')[1].split('```')[0].strip()
        elif '```' in result_text:
            result_text = result_text.split('```')[1].split('```')[0].strip()
        
        start_idx = result_text.find('{')
        end_idx = result_text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            result_text = result_text[start_idx:end_idx+1]
        
        return json.loads(result_text.strip())
    except Exception as e:
        raise Exception(f"Planner error: {str(e)}")

def structurer(data: dict) -> str:
    """Gemma가 영어로 여행 가이드를 생성하도록 하는 함수"""
    prompt = f"""You are an AI that writes detailed travel guides in English. Use the travel information below to create a complete Obsidian-style Markdown guide.

Travel information:
- Destination: {data.get('destination', 'N/A')}
- Duration: {data.get('duration', 'N/A')}
- Preferences: {data.get('preferences', 'N/A')}
- Recommended activities: {', '.join(data.get('activities', []))}
- Tags: {', '.join(data.get('tags', []))}

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
tags: [{', '.join(data.get('tags', []))}]
---

Write the guide in English only."""
    try:
        model = genai.GenerativeModel("gemma-4-31b-it")
        response = model.generate_content(prompt)
        
        result_text = None
        if hasattr(response, 'parts') and response.parts:
            if len(response.parts) > 1 and response.parts[1].text:
                result_text = response.parts[1].text.strip()
            else:
                result_text = response.parts[0].text.strip()
        elif hasattr(response, 'text'):
            result_text = response.text.strip()
        
        if not result_text:
            raise Exception("Structurer 응답이 비어있습니다")
        
        return result_text
    except Exception as e:
        raise Exception(f"Structurer error: {str(e)}")

def visualizer(data: dict) -> str:
    """여행의 시간순 이동 동선을 Mermaid로 시각화"""
    destination = data.get('destination', '목적지')
    activities = data.get('activities', [])
    
    nodes = []
    edges = []
    
    nodes.append('Start["🛬 여행 시작"]')
    
    if '오사카' in destination and '도쿄' in destination:
        nodes.append('A["🏯 오사카 일정"]')
        nodes.append('B["🚄 오사카 -> 도쿄 이동"]')
        nodes.append('C["🌸 도쿄 일정"]')
        nodes.append('End["🏠 귀가"]')
        edges.append('Start --> A')
        edges.append('A --> B')
        edges.append('B --> C')
        edges.append('C --> End')
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

def translate_to_korean(english_markdown: str) -> str:
    """Generate a Korean version of the English markdown while preserving structure and code blocks."""
    prompt = f"""Translate the following English Markdown document into Korean.
- Keep YAML frontmatter structure intact, but translate all text values into Korean.
- Preserve Markdown headings, lists, and formatting.
- Do not translate the contents of code blocks, especially Mermaid code blocks. Keep them as-is.
- Output only the translated Markdown document.

English Markdown:
```markdown
{english_markdown}
```
"""
    try:
        model = genai.GenerativeModel("gemma-4-31b-it")
        response = model.generate_content(prompt)
        
        result_text = None
        if hasattr(response, 'parts') and response.parts:
            if len(response.parts) > 1 and response.parts[1].text:
                result_text = response.parts[1].text.strip()
            else:
                result_text = response.parts[0].text.strip()
        elif hasattr(response, 'text'):
            result_text = response.text.strip()
        
        if not result_text:
            raise Exception("Translation response is empty")
        
        return result_text
    except Exception as e:
        raise Exception(f"Translation error: {str(e)}")

@app.post("/plan_trip")
def plan_trip(trip_query: TripQuery):
    try:
        # 파이프라인 실행
        json_data = planner(trip_query.query)
        english_content = structurer(json_data)
        korean_content = translate_to_korean(english_content)
        mermaid_code = visualizer(json_data)

        # 최종 콘텐츠: 번역된 한국어 마크다운 + Mermaid 다이어그램
        content = f"{korean_content}\n\n## 이동 경로\n\n```mermaid\n{mermaid_code}\n```\n"

        with open("output_travel.md", "w", encoding="utf-8") as f:
            f.write(content)

        return {
            "message": "Gemma 4가 여행 계획을 성공적으로 생성했습니다.",
            "english": english_content,
            "korean": korean_content,
            "mermaid": mermaid_code
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오류 발생: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Codespaces 환경에서는 0.0.0.0 포트 사용이 필수입니다.
    uvicorn.run(app, host="0.0.0.0", port=8000)