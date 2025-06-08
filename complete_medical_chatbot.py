from medical_rag import MedicalRAGSystem
from intent_analyzer import MedicalChat
from emergency_evaluator import EmergencyEvaluator

class CompleteMedicalChat(MedicalChat):
    def __init__(self, rag_system=None):
        # 기존 MedicalChat 초기화
        super().__init__(rag_system)

        # 응급도 평가 추가
        # (응급도 평가가 나중에 만들어서여..)
        self.emergency_evaluator = EmergencyEvaluator()

    def setup_rag_system(self, csv_path="drug_info.csv", chunk_size=300, chunk_overlap=30):

        # RAG 생성
        self.rag = MedicalRAGSystem()

        try:
            # 데이터 로드
            documents = self.rag.load_csv_data(csv_path)
            # 문서 청크 분할
            split_docs = self.rag.split_documents(chunk_size, chunk_overlap)
            # 임베딩 생성 및 벡터스토어 구축
            vectorstore = self.rag.create_vectorstore(split_docs)
            # QA 체인 설정
            qa_chain = self.rag.setup_qa_chain()

            return True
        
        except Exception as e:
            print(f"RAG System failed: {e}")

    # 완전한 의료 상담 프로세스
    def complete_consultation(self, user_question):

        if not self.rag:
            return {"error":"RAG System failed, Check setup_rag_system()"}

        print("상담 시작")
        print(f"사용자 질문 {user_question}")

        try:
            # 의도 분석
            analysis = self.intent_analyzer.analyze_intent(user_question)

            print("분석 결과:")
            print(f"   - 질문 유형: {analysis['query_type']}")
            print(f"   - 감지된 약물: {analysis['detected_drugs']}")
            print(f"   - 증상: {analysis['symptoms']}")
            print(f"   - 응급 키워드: {analysis['emergency_keywords']}")
            print(f"   - 신뢰도: {analysis['confidence']}")

            # 응급도 평가
            emergency_result = self.emergency_evaluator.evaluate_emergency_level(analysis)

            print("응급도 결과:")
            print(f"   - Level: {emergency_result['level']}")
            print(f"   - 설명: {emergency_result['description']}")
            print(f"   - 조치: {emergency_result['action']}")
            print(f"   - 근거: {emergency_result['reasoning']}")

            # RAG 검색 및 답변 생성
            # 기존 enhanced_query 메소드 재활용
            enhanced_query = self.enhance_query(user_question, analysis)
            print(f"강화된 검색어: {enhanced_query}")

            # 기존 RAG 답변 생성
            rag_result = self.rag.ask_question(user_question)
            base_answer = rag_result['result'] if rag_result else "관련 정보를 찾을 수 없습니다."

            # 최종 응답 생성
            response_template = self.emergency_evaluator.get_response(emergency_result)
            final_response = response_template.format(
                base_answer=base_answer,
                level=emergency_result['level'],
                description=emergency_result['description'],
                action=emergency_result['action']
            )

            # 결과 정리
            result = {
                'success': True,
                'question': user_question,
                'analysis': analysis,
                'emergency': emergency_result,
                'base_answer': base_answer,
                'final_response': final_response,
                'enhanced_query': enhanced_query
            }

            # 최종 응답 출력
            print("최종 응답:")
            print(final_response)

            return result
        
        except Exception as e:
            print(f"처리 중 오류 발생: {e}")
            return {
                'success': False,
                'error': str(e),
                'question': user_question
            }
    
    # 테스트 시나리오!
    def demo_conversation(self):
        demo_questions = [
            "타이레놀 먹고 속이 아파요",
            #"해열제 하루에 몇 번 먹어야 하나요?", 
            #"두통에 좋은 약 추천해주세요",
            "호흡곤란이 심해서 119 불러야 하나요?",
            #"활명수는 언제 먹는 거예요?"
        ]
        
        results = []
        
        for i, question in enumerate(demo_questions, 1):
            print(f"\n\n {i}/{len(demo_questions)}")
            result = self.complete_consultation(question)
            results.append(result)
            
            # 간단한 요약 출력 (발표용)
            if result.get('success'):
                analysis = result['analysis']
                emergency = result['emergency']
                print("\n 요약:")
                print(f"   질문 유형: {analysis['query_type']}")
                print(f"   응급도: Level {emergency['level']} - {emergency['description']}")
                
                if emergency['level'] >= 4:
                    print("높은 응급도 감지! 즉시 대응 필요")
                elif emergency['level'] >= 2:
                    print("주의 필요, 경과 관찰")
                else:
                    print("일반 상담")
        
        print(f"\n 완료, 총 {len(results)}개 질문 처리")
        
        # 전체 통계
        emergency_count = sum(1 for r in results if r.get('success') and r['emergency']['level'] >= 4)
        side_effect_count = sum(1 for r in results if r.get('success') and r['analysis']['query_type'] == 'side_effect')
        
        print(f"\n 테스트 통계:")
        print(f"   - 총 질문: {len(results)}개")
        print(f"   - 부작용 상담: {side_effect_count}개")
        print(f"   - 응급상황 감지: {emergency_count}개")
        
        return results
    
    def quick_test(self, question):
        """빠른 단일 질문 테스트"""
        
        if not self.rag:
            print("RAG 시스템을 먼저 구축해주세요: chatbot.setup_rag_system()")
            return None
        
        print(f" 빠른 테스트: '{question}'")
        result = self.complete_consultation(question)
        
        if result.get('success'):
            print("처리 완료!")
            return result
        else:
            print(f"처리 실패: {result.get('error')}")
            return None

def main():
    # 쳇봇 초기화
    chat = CompleteMedicalChat()

    # RAG 구축
    success = chat.setup_rag_system('drug_info.csv')

    if not success:
        print("rag 구축 실패")
        return None
    
    # 테스트 실행
    results = chat.demo_conversation()
    
    return chat

def single_test():
    
    print("응급상황 감지 테스트")
    
    chat = CompleteMedicalChat()
    
    if chat.setup_rag_system('drug_info.csv'):
        # 응급상황 테스트
        emergency_question = "타이레놀 먹고 호흡곤란이 심하고 의식이 흐려져요"
        result = chat.quick_test(emergency_question)
        
        if result and result.get('success'):
            level = result['emergency']['level']
            if level >= 4:
                print(f"\n 응급상황 감지 테스트 성공! Level {level}")
            else:
                print(f"\n일반 상담으로 분류됨: Level {level}")
        
        return chat
    else:
        print("시스템 구축 실패")
        return None

if __name__ == "__main__":

    chatbot_system = main()

    
    # chatbot_system = single_test()