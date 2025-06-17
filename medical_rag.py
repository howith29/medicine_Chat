import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
import re
import time
import hashlib

class MedicalRAGSystem:
    def __init__(self):
        self.documents = []
        self.vectorstore = None
        self.query_cache = {}
        self.cache_stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

    def load_csv_data(self, path):
        df = pd.read_csv(path)
        documents = []
        
        for idx, row in df.iterrows():
            drug_name = row['제품명']
            
            field_mappings = {
                '효능효과': '효능효과',
                '사용법': '사용법',
                '주의사항': '주의사항',
                '상호작용': '상호작용',
                '부작용': '부작용',
                '보관법': '보관법'
            }
            
            for field, label in field_mappings.items():
                content = row.get(field)
                if pd.notna(content):
                    doc = Document(
                        page_content=f"{drug_name}의 {label}: {content}",
                        metadata={
                            'drug_name': drug_name,
                            'field': label,
                            'item_code': row['품목기준코드'],
                            'source_row': idx
                        }
                    )
                    documents.append(doc)
        
        self.documents = documents
        return documents

    def split_documents(self, chunk_size=500, chunk_overlap=50):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        split_docs = splitter.split_documents(self.documents)
        return split_docs

    def create_vectorstore(self, split_docs):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.from_documents(split_docs, embedding=embeddings)
        self.vectorstore = vectorstore
        return vectorstore

    def retrieve_documents(self, query, k=4):
        if not self.vectorstore:
            print("Vectorstore not initialized.")
            return []
        retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(query)
        return docs

    def generate_answer_by_type(self, question, retrieved_docs, query_type='other'):
        llm = ChatOpenAI(model="gpt-4", temperature=0.3)
        
        combined_docs = "\n".join([doc.page_content for doc in retrieved_docs])
        
        prompts = {
            'side_effect': f"""
당신은 전문적인 의료 상담 챗봇입니다. 부작용 관련 질문에 대해 다음 형식으로 답변해주세요:

**답변 형식:**
[약물과 증상 관계 설명 2-3문장]
[부작용 발생 가능성과 일반적 대처법 2-3문장]  
[모니터링 포인트나 추가 주의사항 1-2문장]

**답변 원칙:**
- 의학적으로 정확하고 신중하게 답변
- 부작용의 심각성을 적절히 전달
- 응급상황 판단은 별도 시스템에서 처리하므로 "즉시 병원" 같은 응급 지시 제외
- 불확실한 경우 의료진 상담 권유

참고정보:
{combined_docs}

사용자 질문: "{question}"

답변:
            """,
            
            'usage': f"""
당신은 전문적인 의료 상담 챗봇입니다. 복용법 관련 질문에 대해 다음 형식으로 답변해주세요:

**답변 형식:**
[기본 복용법과 용량 정보 2-3문장]
[복용 시 주의사항이나 금기 2-3문장]
[복용 효과 최적화를 위한 팁이나 추가 안내 1-2문장]

**답변 원칙:**
- 정확한 용법·용량 정보 제공
- 안전한 복용을 위한 주의사항 강조
- 개인차가 있을 수 있음을 안내
- 처방약의 경우 의사 지시 우선임을 명시

참고정보:
{combined_docs}

사용자 질문: "{question}"

답변:
            """,
            
            'efficacy': f"""
당신은 전문적인 의료 상담 챗봇입니다. 약물 정보/효능 관련 질문에 대해 다음 형식으로 답변해주세요:

**답변 형식:**
[약물의 주요 효능과 작용기전 2-3문장]
[적응증과 사용 가능한 상황 2-3문장]
[선택 시 고려사항이나 대안 1-2문장]

**답변 원칙:**
- 객관적이고 균형잡힌 정보 제공
- 약물의 한계와 부작용 가능성도 언급
- 개인의 건강상태에 따른 적합성 차이 안내
- 전문의 상담의 중요성 강조

참고정보:
{combined_docs}

사용자 질문: "{question}"

답변:
            """,
            
            'other': f"""
당신은 친절한 의료 상담 챗봇입니다. 일반적인 의료 질문에 대해 다음 형식으로 답변해주세요:

**답변 형식:**
[질문에 대한 기본적인 의학 정보 2-3문장]
[일반적인 관리 방법이나 주의사항 2-3문장]
[전문의 상담이 필요한 경우나 추가 안내 1-2문장]

**답변 원칙:**
- 이해하기 쉬운 언어로 설명
- 일반적인 의학 상식 수준에서 답변
- 진단이나 처방은 할 수 없음을 명시
- 적절한 의료진 상담 권유

참고정보:
{combined_docs}

사용자 질문: "{question}"

답변:
            """
        }
        
        # 해당 타입의 프롬프트 선택 (기본값: other)
        prompt = prompts.get(query_type, prompts['other'])
        
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"답변 생성 오류: {e}")
            return "죄송합니다. 답변 생성 중 오류가 발생했습니다. 다시 시도해주세요."

    def generate_answer(self, question, retrieved_docs):
        """기존 방식의 답변 생성 (하위 호환성 유지)"""
        return self.generate_answer_by_type(question, retrieved_docs, 'other')
        
    def ask_question(self, question, query_type='other'):
        start_time = time.time()
        self.cache_stats['total_queries'] += 1
        cache_key = self.generate_cache_key(question)

        if cache_key in self.query_cache:
            self.cache_stats['cache_hits'] += 1
            cached_result = self.query_cache[cache_key]
            return cached_result

        self.cache_stats['cache_misses'] += 1

        retrieved_docs = self.retrieve_documents(question)
        answer = self.generate_answer_by_type(question, retrieved_docs, query_type)
        
        result = {
            'result': answer,
            'source_documents': retrieved_docs
        }

        self.save_to_cache(cache_key, result)
        elapsed_time = time.time() - start_time
        print(f"처리시간: {elapsed_time:.3f}초, 캐시율: {self.get_cache_hit_rate():.1f}%")
        return result

    def generate_cache_key(self, question):
        normalized = re.sub(r'\s+', ' ', question.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()[:12]

    def save_to_cache(self, cache_key, result):
        max_cache_size = 100
        if len(self.query_cache) >= max_cache_size:
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
        self.query_cache[cache_key] = result

    def get_cache_hit_rate(self):
        total = self.cache_stats['total_queries']
        if total == 0:
            return 0.0
        return (self.cache_stats['cache_hits'] / total) * 100

    def get_cache_stats(self):
        return {**self.cache_stats, 'cache_hit_rate': f"{self.get_cache_hit_rate():.1f}%", 'cached_queries': len(self.query_cache)}

    def clear_cache(self):
        self.query_cache.clear()
        self.cache_stats = {'total_queries': 0, 'cache_hits': 0, 'cache_misses': 0}
        print("캐시 초기화 완료")
