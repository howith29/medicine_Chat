import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import re
import time
import hashlib


class MedicalRAGSystem:
    def __init__(self):
        self.documents = []
        self.vectorstore = None
        self.qa_chain = None

        # 캐싱 시스템
        self.query_cache = {}
        self.cache_stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

    def load_csv_data(self, path):
        # 데이터 로드
        df = pd.read_csv(path)
        documents = []

        for idx, row in df.iterrows():
            drug_text = self.combine_drug_info(row)

            doc = Document(
                page_content=drug_text,
                metadata={
                    'drug_name': row['제품명'],
                    'company': row['업체명'],
                    'item_code': row['품목기준코드'],
                    'source_row': idx
                }
            )
            documents.append(doc)

        self.documents = documents
        return documents


    # 약물 정보를 하나의 텍스트로 결합
    def combine_drug_info(self, row):
        parts = []

        # 제품명
        parts.append(f"약물명: {row['제품명']}")

                # 업체명
        if pd.notna(row['업체명']):
            parts.append(f"제조사: {row['업체명']}")
        
        # 효능효과
        if pd.notna(row['효능효과']):
            parts.append(f"효능효과: {row['효능효과']}")
        
        # 사용법
        if pd.notna(row['사용법']):
            parts.append(f"사용법: {row['사용법']}")
        
        # 주의사항  
        if pd.notna(row['주의사항']):
            parts.append(f"주의사항: {row['주의사항']}")
        
        # 상호작용
        if pd.notna(row['상호작용']):
            parts.append(f"상호작용: {row['상호작용']}")
        
        # 부작용
        if pd.notna(row['부작용']):
            parts.append(f"부작용: {row['부작용']}")
        
        # 보관법
        if pd.notna(row['보관법']):
            parts.append(f"보관법: {row['보관법']}")
        
        return "\n".join(parts)
    
    # 문서 -> 청크 분할
    def split_documents(self, chunk_size=1000, chunk_overlap=200):
    
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function = len,
            separators = ["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

        split_docs = splitter.split_documents(self.documents)

        return split_docs

    # 임베딩 생성 및 FAISS 벡터스토어 구측
    def create_vectorstore(self, split_docs):

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        vectorstore = FAISS.from_documents(
            documents=split_docs,
            embedding=embeddings
        )

        self.vectorstore = vectorstore
        return vectorstore
    
    # QA chain
    def setup_qa_chain(self):

        llm = ChatOpenAI(
            model = "gpt-4",
            temperature=0.3 # 의료 정보이므로 낮은 창의성
        )

        retriever = self.vectorstore.as_retriever(
            search_type = "similarity",
            search_kwargs = {"k":3} # 상위 3개 청크 검색
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm = llm, 
            chain_type="stuff",
            retriever = retriever,
            return_source_documents = True, # 참고 문서 반환
            verbose = True
        )

        self.qa_chain = qa_chain
        return qa_chain

    # 검색 테스트
    def search_test(self, query):
        
        if not self.vectorstore:
            print("백터스토어 존재하지 않음")
            return
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k":3})
        docs = retriever.get_relevant_documents(query)

        for i, doc in enumerate(docs, 1):
            print(f"\n{i}. 약물: {doc.metadata.get('drug_name', 'Unknown')}")
            print(f"   내용: {doc.page_content[:200]}...")

    # 질문 (캐싱 적용)
    def ask_question(self, question):

        start_time = time.time()

       # 캐시 통계 업데이트
        self.cache_stats['total_queries'] += 1

        # 캐시 확인
        cache_key = self.generate_cache_key(question)

        if cache_key in self.query_cache:
            self.cache_stats['cache_hits'] += 1
            cached_result = self.query_cache[cache_key]

            print("답변: ")
            print(cached_result['result'])
            print(f"\n 참고 문서: ")
            for i, doc in enumerate(cached_result['source_documents'], 1):
                drug_name = doc.metadata.get('drug_name', 'Unknown')
                print(f"   {i}. {drug_name}: {doc.page_content[:100]}...")
            
            return cached_result
        
        # 캐시 미스
        self.cache_stats['cache_misses'] += 1

        if not self.qa_chain:
            print("QA 체인이 준비되지 않음")
            return None

        result = self.qa_chain({"query": question})

        self.save_to_cache(cache_key, result)

        # 결과 출력
        print("답변: ")
        print(result['result'])

        print(f"\n 참고 문서: ")
        for i, doc in enumerate(result['source_documents'], 1):
            drug_name = doc.metadata.get('drug_name', 'Unknown')
            print(f"   {i}. {drug_name}: {doc.page_content[:100]}...")

        elapsed_time = time.time() - start_time
        print(f"\n 처리 시간: {elapsed_time:.3f}초")
        print(f"캐시 적중률: {self.get_cache_hit_rate():.1f}%")
        return result
    
    # 캐시 키 생성
    def generate_cache_key(self, question):
        
        # 질문 정규화
        normalized = re.sub(r'\s+',' ',question.lower().strip())
        # 해시 생성
        return hashlib.md5(normalized.encode()).hexdigest()[:12]
    
    # 크기 제한 캐시 저장
    def save_to_cache(self, cache_key, result):
        max_cache_size = 100 # 캐시 최대 크기 100개

        if len(self.query_cache) >= max_cache_size:
            # 가장 오래된 항목 제거(FIFO)
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
        
        self.query_cache[cache_key] = result
    
    # 캐시 적중률 계산
    def get_cache_hit_rate(self):
        
        total = self.cache_stats['total_queries']
        if total == 0:
            return 0.0
        return (self.cache_stats['cache_hits'] / total) * 100

    # 캐시 통계 반환
    def get_cache_stats(self):
        return {
            **self.cache_stats,
            'cache_hit_rate': f"{self.get_cache_hit_rate():.1f}%",
            'cached_queries': len(self.query_cache),
        }
    
    # 캐시 초기화
    def clear_cache(self):
        self.query_cache.clear()
        self.cache_stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        print("캐시가 초기화되었습니다.")

# 메인 실행 함수
def main():

    rag = MedicalRAGSystem()

    try:
        documents = rag.load_csv_data('drug_info.csv')
        split_docs = rag.split_documents(chunk_size=300, chunk_overlap=30)
        vectorstore = rag.create_vectorstore(split_docs)
        qa_chain = rag.setup_qa_chain()

        # 검색 테스트
        # print("기본 검색 테스트:")
        # test_queries = [
        #     "타이레놀 부작용",
        #     "소화제 복용법", 
        #     "해열제 효능"
        # ]

        # for query in test_queries:
        #     rag.search_test(query)

        # QA 테스트
        print("질의응답 테스트")
        qa_test_questions = [
            "활명수 먹고 속이 아프면 어떻게 해야 하나요?",
            "소화불량에 좋은 약 추천해주세요", 
            "활명수 먹고 속이 아프면 어떻게 해야 하나요?", # 중복 질문 테스트
        ]
        
        for i, question in enumerate(qa_test_questions, 1):
            print(f"\n{'='*20} 질문 {i} {'='*20}")
            print(f"질문: {question}")
            print("-" * 50)
            rag.ask_question(question)

        # 캐시 통계 출력
        stats = rag.get_cache_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")

        return rag 

    except FileNotFoundError:
        print("파일이 없습니다.")
        return None
    except Exception as e:
        print(f"오류 발생:{str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    rag_system = main()

