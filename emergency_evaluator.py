
class EmergencyEvaluator:
    def __init__(self):
        
        self.emergency_levels ={
            5: {
                'keywords': [
                    '호흡곤란', '숨쉬기어려움', '숨이막힘', '질식',
                    '의식잃음', '기절', '쓰러짐', '의식불명',
                    '경련', '발작', '간질', 
                    '심한가슴통증', '가슴을쥐어짬', '심장마비',
                    '토혈', '혈변', '대량출혈', '심한출혈',
                    '119', '응급실', '응급차'
                ],
                'action': "즉시 119 신고 또는 응급실 방문",
                'description': "생명을 위협할 수 있는 응급상황"
            },
            
            # Level 4: 긴급 병원 방문 (당일 내)
            4: {
                'keywords': [
                    '심한통증', '극심한통증', '참을수없는통증',
                    '고열', '심한발열', '40도', '고온',
                    '심한어지러움', '극심한어지러움',
                    '심한구토', '계속구토', '멈추지않는구토',
                    '심한설사', '혈성설사',
                    '시야흐림', '시야장애', '눈이보이지않음',
                    '심한두통', '극심한두통', '머리가깨질것같음'
                ],
                'action': "당일 내 병원 응급실 방문 필요",
                'description': "빠른 의료진 진료가 필요한 상황"
            },
            
            # Level 3: 빠른 병원 방문 (1-2일 내)
            3: {
                'keywords': [
                    '지속적인통증', '계속되는통증', '악화되는',
                    '발열', '열이남', '몸이뜨거움',
                    '심한속쓰림', '극심한속쓰림',
                    '지속적구토', '반복적구토',
                    '발진악화', '부종심함', '붓기심함',
                    '호흡이상', '숨이가쁨'
                ],
                'action': "1-2일 내 병원 방문 권장",
                'description': "지속되거나 악화되는 증상"
            },
            
            # Level 2: 경과 관찰 후 병원 방문
            2: {
                'keywords': [
                    '속쓰림', '위장장애', '소화불량', '복통',
                    '두통', '어지러움', '메스꺼움', '구토',
                    '설사', '변비', '복부팽만',
                    '피부발진', '가려움', '두드러기',
                    '피로감', '무력감', '식욕부진'
                ],
                'action': "2-3일 경과 관찰 후 지속되면 병원 방문",
                'description': "일반적인 부작용, 경과 관찰 필요"
            },
            
            # Level 1: 일반적 부작용 (자가 관리)
            1: {
                'keywords': [
                    '약간의', '살짝', '조금', '미미한',
                    '가벼운두통', '가벼운어지러움',
                    '약간의속쓰림', '가벼운메스꺼움'
                ],
                'action': "복용 중단 후 자가 관찰",
                'description': "경미한 증상, 대부분 자연 회복"
            }
        }

    # 의도분석 결과를 바탕으로 응급도 평가 
    def evaluate_emergency_level(self, analysis_result):

        if analysis_result.get('query_type') != 'side_effect':
            return {
                'level':0,
                'action': "일반 상담",
                'description':'응급도 평가 대상 아님',
                'reasoning':'부작용 상담이 아닙니다',
                'matched_keywords':[]
            }
        
        text_to_analyze = ' '.join([
            ' '.join(analysis_result.get('symptoms',[])),
            ' '.join(analysis_result.get('emergency_keywords',[]))
        ]).lower()

        print(f"응급도 분석 대상 텍스트: {text_to_analyze}")

        # 레벨별 점수 계산
        # 가장 많은 키워드가 매칭된 레벨 = 해당 응급도
        level_scores = {}
        matched_keywords = {}

        for level, info in self.emergency_levels.items():
            score = 0
            matched = []

            for keyword in info['keywords']:
                if keyword.lower() in text_to_analyze:
                    score += 1
                    matched.append(keyword)

            level_scores[level] = score
            matched_keywords[level] = matched

            print(f"레벨 점수: {level_scores}")

            max_level = 1
            max_score = 0

            # 가장 높은 레벨 선택
            for level, score in level_scores.items():
                if score > max_score:
                    max_score = score
                    max_level = level

            # 매칭 키워드 없는 경우 level 2 (일반 부작용)
            if max_score == 0:
                max_level = 2
                reasoning = "일반적인 부작용으로 추정"
                final_matched = []
            else:
                reasoning = f"Level {max_level} 키워드 매칭: {matched_keywords[max_level]}"
                final_matched = matched_keywords[max_level]

            result = {
                'level': max_level,
                'action': self.emergency_levels[max_level]['action'],
                'description': self.emergency_levels[max_level]['description'],
                'reasoning': reasoning,
                'matched_keywords': final_matched
            }
            print(f"응급도 평가 결과: Level {max_level}")

            return result
        
        # 응급도에 따른 응답
        def get_response(self, emergency_result):
            level = emergency_result['level']

            if level >= 5:
                return """
                 응급 상황 감지
                 {base_answer}
                 응급도: Level {level} - {description}
                {action}

                - 119 신고 또는 가장 가까운 응급실 방문
                - 약물 복용 즉시 중단
                - 증상 변화 주의 깊게 관찰

                주의: 이는 의료진의 진단을 대체할 수 없습니다.
                """
            elif level >=4:
                return """
                주의 필요
                {base_answer}
                응급도: Level {level} - {description}
                {action}

                - 약물 복용 즉시 중단
                - 당일 내 병원 응급실 또는 응급진료 방문
                - 증상 악화 시 즉시 119 신고

                주의: 이는 의료진의 진단을 대체할 수 없습니다. 증상이 악화되면 즉시 응급실을 방문하세요.

                """
            elif level >= 3:
                return """
                병원 방문 권장
                {base_answer}
                응급도: Level {level} - {description}
                {action}

                - 약물 복용 중단 고려
                - 1-2일 내 병원 방문
                - 증상 지속/악화 시 더 빨리 방문
                주의: 이는 의료진의 진단을 대체할 수 없습니다. 증상이 심해지면 더 빨리 의료진과 상담하세요.
                """
            
            elif level >= 2:
                return """
                경과 관찰
                {base_answer}
                응급도: Level {level} - {description}
                {action}

                - 약물 복용 일시 중단
                - 충분한 수분 섭취
                - 2-3일 경과 관찰
                - 증상 지속 시 병원 방문
                주의: 이는 의료진의 진단을 대체할 수 없습니다. (참고) 대부분 시간이 지나면 호전됩니다.

                """
            
            else:
                return """
                일반 상담
                {base_answer}
                응급도: Level {level} - {description}
                {action}

                - 증상 경과 관찰
                - 필요시 의료진 상담

                (참고) 일반적인 의료 상담입니다.
               """


# 응급도 평가 테스트
def test_emergency_evaluation():
    
    evaluator = EmergencyEvaluator()
    
    # 테스트 케이스들 (다양한 응급도)
    test_cases = [
        {
            'query_type': 'side_effect',
            'symptoms': ['호흡곤란', '의식잃음'],
            'emergency_keywords': ['119'],
            'expected_level': 5
        },
        {
            'query_type': 'side_effect', 
            'symptoms': ['심한복통', '토혈'],
            'emergency_keywords': [],
            'expected_level': 5
        },
        {
            'query_type': 'side_effect',
            'symptoms': ['지속적인두통', '발열'],
            'emergency_keywords': [],
            'expected_level': 3
        },
        {
            'query_type': 'side_effect',
            'symptoms': ['속쓰림', '메스꺼움'],
            'emergency_keywords': [],
            'expected_level': 2
        },
        {
            'query_type': 'usage',  # 부작용 아님
            'symptoms': ['두통'],
            'emergency_keywords': [],
            'expected_level': 0
        }
    ]
    
    print("응급도 평가 테스트")
    
    correct_count = 0
    total_count = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n테스트 {i}:")
        print(f"입력: {case}")
        
        result = evaluator.evaluate_emergency_level(case)
        
        print(f"결과: Level {result['level']} - {result['description']}")
        print(f"조치: {result['action']}")
        print(f"근거: {result['reasoning']}")
        
        # 예상 결과와 비교
        expected = case['expected_level']
        actual = result['level']
        
        if actual == expected:
            print(f"정확! (예상: Level {expected})")
            correct_count += 1
        else:
            print(f"불일치 (예상: Level {expected}, 실제: Level {actual})")
        
        print("-" * 30)
    
    print(f"\n 테스트 결과: {correct_count}/{total_count} 정확 ({correct_count/total_count*100:.1f}%)")

if __name__ == "__main__":
    test_emergency_evaluation()