#!/usr/bin/env python
"""
CLI 인터페이스: 터미널에서 여행 계획 생성
"""

import sys
sys.path.insert(0, '/workspaces/thx_air')

from main import planner, structurer, visualizer, detect_language
import json

def main():
    print("\n" + "="*60)
    print("🌍 AI 여행 플래너 - 터미널 인터페이스")
    print("="*60)
    
    # 사용자 입력
    query = input("\n📝 여행 계획에 대해 물어봐주세요 (예: '일본 도쿄 5일, 미식 투어'): ").strip()
    budget = input("💰 예산이 있다면 입력해주세요 (예: 인당 100만원, 200000엔): ").strip() or None
    people_raw = input("👥 몇 명이 여행하나요? (숫자만 입력, 예: 2): ").strip()
    people = None
    if people_raw:
        try:
            people = int(people_raw)
            if people <= 0:
                raise ValueError
        except ValueError:
            print("❌ people은 1 이상의 숫자여야 합니다.")
            return

    if not query:
        print("❌ 질문을 입력해주세요!")
        return
    
    print("\n⏳ Gemma 4 모델이 여행 계획을 생성 중입니다...\n")
    
    try:
        # 1단계: Planner
        print("[1/3] 📋 여행 정보 추출 중...")
        json_data = planner(query)
        if budget:
            json_data['budget'] = budget
        if people is not None:
            json_data['people'] = people
        print(f"✅ 추출 완료: {json_data}\n")
        
        # 2단계: 언어 감지 및 Structurer
        print("[2/3] 📝 여행 가이드 생성 중...")
        language = detect_language(query)
        print(f"감지된 언어: {language}")
        markdown_content = structurer(json_data, language)
        print("✅ 가이드 생성 완료\n")
        
        # 3단계: Visualizer
        print("[3/3] 🗺️ 이동 경로 생성 중...")
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
        if budget:
            print(f"💰 예산: {budget}")
        if people is not None:
            print(f"👥 인원: {people}명")
        print(f"🌐 언어: {language}")
        print("\n✅ 파일이 저장되었습니다!\n")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        return

if __name__ == "__main__":
    main()
