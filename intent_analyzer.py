import json
from langchain_openai import ChatOpenAI
from medical_rag import MedicalRAGSystem

class MedicalIntentAnalyzer:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1
        )
    
    # 응급 키워드 리스트
        self.emergency_keywords = [
            '호흡곤란', '숨쉬기어려움', '의식잃음', '기절', '쓰러짐',
            '경련', '발작', '심한복통', '심한두통', '가슴아픔', 
            '토혈', '혈변', '대량출혈', '119', '응급실',
            '심각', '위험', '생명', '즉시', '급하게'    
        ]

    # 질문 의도 파악
    def analyze_intent(self, user_message):
        
        prompt = f"""
        사용자의 의료 관련 질문을 분석하여 JSON 형태로 응답해줘
        사용자 질문: "{user_message}"
        다음 형식으로 분석해:
        {{
            "query_type":"side_effect|usage|efficacy|other",
            "detected_drugs":["약물명1","약물명2"],
            "symptoms":["증상1","증상2"],
            "emergency_keywords":["응급키워드1"],
            "confidence": 0.85,
            "reasoning":"분류 이유"
        }}
        query_type 분류 기준:
        - side_effect: 부작용, 이상반응, "먹고 아파요", "복용 후 증상" 등
        - usage: 복용법, 사용법, "어떻게 먹어야", "몇 번", "언제" 등  
        - efficacy: 효능, 효과, "어떤 약", "추천", "좋은 약", "무슨 약", "약 먹어야" 등
        - other: 위에 해당하지 않는 경우
        """

        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response.content)
            
            # 응급 키워드 추가
            additional_emergency = self.detect_emergency_keywords(user_message)
            if additional_emergency:
                result["emergency_keywords"].extend(additional_emergency)
                result["emergency_keywords"] = list(set(result["emergency_keywords"]))
            
            return result
        
        except Exception as e:
            print(f"의도 분석 오류: {e}")
            return {    
                "query_type":"other",
                "detected_drugs": [],
                "symptoms": [],
                "emergency_keywords": [],
                "confidence": 0.0,
                "reasoning":"분석 실패"
            }
    
    # 응급 키워드 감지
    def detect_emergency_keywords(self, text):

        found_keywords = []
        text_lower = text.lower()

        for keyword in self.emergency_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)

        return found_keywords

# 기존 RAG + 의도 분석 
class MedicalChat:
    def __init__(self, rag_system):
        self.rag = rag_system
        self.intent_analyzer = MedicalIntentAnalyzer()
    
    # 질문 파이프라인
    def process_question(self, user_question):

        # 의도 분석
        analysis = self.intent_analyzer.analyze_intent(user_question)

        print("분석 결과:")
        print(f"   - 질문 유형: {analysis['query_type']}")
        print(f"   - 감지된 약물: {analysis['detected_drugs']}")
        print(f"   - 증상: {analysis['symptoms']}")
        print(f"   - 응급 키워드: {analysis['emergency_keywords']}")
        print(f"   - 신뢰도: {analysis['confidence']}")

        # 검색어 강화
        enhanced_query = self.enhance_query(user_question, analysis)

        # 답변 생성
        result = self.rag.ask_question(user_question)

        return result, analysis
    
    # 분석 결과로 검색어 강화
    def enhance_query(self, original_query, analysis):

        enhanced_parts = [original_query]

        # 약물명 추가
        if analysis["detected_drugs"]:
            enhanced_parts.extend(analysis["detected_drugs"])

        # 증상 추가  
        if analysis["symptoms"]:
            enhanced_parts.extend(analysis["symptoms"])

       # 질문 유형별 키워드 추가
        query_type = analysis["query_type"]
        if query_type == "side_effect":
            enhanced_parts.append("부작용")
        elif query_type == "usage":
            enhanced_parts.append("복용법 사용법")
        elif query_type == "efficacy":
            enhanced_parts.append("효능 효과")
        
        return " ".join(enhanced_parts)
    
    # 검색 수행
    def search_only(self, query):
        analysis = self.intent_analyzer.analyze_intent(query)
        enhanced_query = self.enhance_query(query, analysis)

        print(f"검색: {enhanced_query}")
        self.rag.search_test(enhanced_query)

        return analysis

# 테스트   
def main():
    rag_system = MedicalRAGSystem()

    try:
        documents = rag_system.load_csv_data('drug_info.csv')
        split_docs = rag_system.split_documents(chunk_size=300, chunk_overlap=30)
        vectorstore = rag_system.create_vectorstore(split_docs)
        qa_chain = rag_system.setup_qa_chain()

        chat = MedicalChat(rag_system)

        test_questions = [
            "타이레놀 먹고 속이 아파요",
            "해열제 하루에 몇 번 먹어야 하나요?",
            "두통에 좋은 약 추천해주세요",
            "활명수는 언제 먹는 거예요?"
        ]

        for question in test_questions:
            print(f"테스트 질문: {question}")
            
            # 전체 파이프라인 실행
            result, analysis = chat.process_question(question)
            
            print("최종 결과:")
            print(f"   - 질문 유형: {analysis['query_type']}")
            print(f"   - 응급도: {'⚠️ 응급키워드 감지' if analysis['emergency_keywords'] else '일반 상담'}")
        
        return chat
        
    except Exception as e:
        print(f"오류 발생: {e}")
        return None

def quick_test():
    analyzer = MedicalIntentAnalyzer()
    
    test_questions = [
        "타이레놀 먹고 속이 아파요",
        "해열제 복용법 알려주세요",
        "두통약 추천해주세요",
        "호흡곤란이 심해요"
    ]
    
    print("의도분석 빠른 테스트")
    for question in test_questions:
        analysis = analyzer.analyze_intent(question)
        print(f"\n'{question}' → {analysis['query_type']}")
        if analysis['emergency_keywords']:
            print(f"  ⚠️ 응급: {analysis['emergency_keywords']}")

if __name__ == "__main__":
    # 빠른 테스트
    # quick_test()

    chat = main()
    