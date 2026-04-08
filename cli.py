#!/usr/bin/env python
"""
CLI 인터페이스: 터미널에서 여행 계획 생성
"""

import sys
sys.path.insert(0, '/workspaces/thx_air')

from main import planner, structurer, translate_to_korean, visualizer
import json

def main():
    print("\n" + "="*60)
    print("🌍 AI 여행 플래너 - 터미널 인터페이스")
    print("="*60)
    
    # 사용자 입력
    query = input("\n📝 여행 계획에 대해 물어봐주세요 (예: '일본 도쿄 5일, 미식 투어'): ").strip()
    
    if not query:
        print("❌ 질문을 입력해주세요!")
        return
    
    print("\n⏳ Gemma 4 모델이 여행 계획을 생성 중입니다...\n")
    
    try:
        # 1단계: Planner
        print("[1/4] 📋 여행 정보 추출 중...")
        json_data = planner(query)
        print(f"✅ 추출 완료: {json_data}\n")
        
        # 2단계: Structurer (English)
        print("[2/4] 📝 English travel guide 생성 중...")
        english_content = structurer(json_data)
        print("✅ English guide 생성 완료\n")
        
        # 3단계: 한국어 번역
        print("[3/4] 🌐 한국어 번역 중...")
        markdown_content = translate_to_korean(english_content)
        print("✅ 한국어 번역 완료\n")
        
        # 4단계: Visualizer
        print("[4/4] 🗺️ 이동 경로 생성 중...")
        mermaid_code = visualizer(json_data)
        print("✅ 경로 생성 완료\n")
        
        # 최종 파일 저장
        content = f"{markdown_content}\n\n## 이동 경로\n\n```mermaid\n{mermaid_code}\n```\n"
        
        with open("output_travel.md", "w", encoding="utf-8") as f:
            f.write(content)
        
        print("="*60)
        print("✨ 여행 계획이 완성되었습니다!")
        print("="*60)
        print("\n📄 파일: output_travel.md")
        print(f"📍 목적지: {json_data.get('destination', 'N/A')}")
        print(f"⏰ 기간: {json_data.get('duration', 'N/A')}")
        print(f"❤️  취향: {json_data.get('preferences', 'N/A')}")
        print("\n✅ 파일이 저장되었습니다!\n")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        return

if __name__ == "__main__":
    main()
