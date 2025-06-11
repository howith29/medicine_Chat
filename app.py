from flask import Flask, render_template, request, jsonify
import traceback
import sys


# 기존 의료 챗봇 모듈 import
try:
    from complete_medical_chatbot import CompleteMedicalChat
    from medical_rag import MedicalRAGSystem
except ImportError as e:
    print(f"Import Error: {e}")


app = Flask(__name__)
app.secret_key = 'yakTalk_medical_chatbot_2025'

# 전역 챗봇 인스턴스
chatbot = None

# 챗봇 시스템 초기화
def init_chatbot():

    global chatbot

    try:
        # 챗봇 인스턴스 생성
        chatbot = CompleteMedicalChat()
        # RAG 시스템 구축
        success = chatbot.setup_rag_system('drug_info.csv')

        if success:
            print("initialize chatbot done")
            return True
        else:
            print("initialize chatbot failed")
            return False
    except Exception as e:
        print(f"initialize chatbot failed: {e}")
        traceback.print_exc()
        return False

# 메인 페이지
@app.route('/')
def index():
    return render_template('index.html')

# 채팅 api endpoint
@app.route('/api/chat', methods=['POST'])
def chat_api():
    if not chatbot:
        return jsonify({
            'success': False,
            'error':'챗봇 시스템 초기화 되지 않음'
        }), 500
    
    try:
        # 사용자 입력 받기
        data = request.get_json()
        user_message = data.get('message','').strip()

        if not user_message:
            return jsonify({
                'success':False,
                'error':'메시지를 입력해주세요'
            }), 400
        print(f"사용자 질문: {user_message}")

        # chatbot 처리
        result = chatbot.complete_consultation(user_message)

        if result.get('success'):
            # 응답 성공
            analysis = result['analysis']
            emergency = result['emergency']

            # 응급도별 색상 결정
            emergency_color = get_emergency_color(emergency['level'])

            response_data = {
                'success': True,
                'message': result['final_response'],
                'analysis': {
                    'query_type': analysis['query_type'],
                    'detected_drugs': analysis['detected_drugs'],
                    'symptoms': analysis['symptoms'],
                    'confidence': analysis['confidence']
                },
                'emergency':{
                    'level': emergency['level'],
                    'description': emergency['description'],
                    'action': emergency['action'],
                    'color': emergency_color
                }
            }
            print(f"응답 생성 완료(응급도: level{emergency['level']})")
            return jsonify(response_data)

        else:
            # 처리 실패
            return jsonify({
                'success': False,
                'error': result.get('error','처리 중 오류 발생')
            }), 500
        
    except Exception as e:
        print(f"API 처리 오류: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error':'서버 내부 오류 발생'
        }), 500

# 응급도별 색상 반환
def get_emergency_color(level):
    colors = {
        0: '#6c757d',  # 회색 (일반 상담)
        1: '#28a745',  # 초록 (경미)
        2: '#ffc107',  # 노랑 (경과 관찰)
        3: '#fd7e14',  # 주황 (병원 방문)
        4: '#dc3545',  # 빨강 (긴급)
        5: '#8b0000'   # 진한 빨강 (생명 위험)
    }
    return colors.get(level, '#6c757d') # 회색

# 시스템 상태 체크
@app.route('/api/health')
def health_check():
    if chatbot:
        return jsonify({
            'status':'healthy',
            'message':'시스템이 정상 작동중'
        })
    else:
        return jsonify({
            'status':'unhealthy',
            'message':'챗봇 시스템 초기화 실패'
        }), 503

if __name__ == '__main__':
    
    # 챗봇 시스템 초기화
    if init_chatbot():

        # flask app 실행
        app.run(
            host='127.0.0.1',
            port=5000,
            debug=True,
            threaded=True
        )
    else:
        print("챗봇 시스템 초기화 실패")
        sys.exit(1)