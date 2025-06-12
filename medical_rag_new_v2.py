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

    def generate_answer(self, question, retrieved_docs):
        llm = ChatOpenAI(model="gpt-4", temperature=0.5)
        
        combined_docs = "\n".join([doc.page_content for doc in retrieved_docs])
        
        prompt = f"""
너는 매우 친절하고 상세한 의료 상담 챗봇이야. 
아래 제공된 참고 정보를 바탕으로 사용자의 질문에 대해 최대한 상세하고 친절하게 설명해줘. 
만약 참고 정보에 없는 내용이라면 아는 범위에서 일반적인 조언을 제공해.

참고정보:
{combined_docs}

사용자 질문: "{question}"

답변:
"""
        
        response = llm.invoke(prompt)
        return response.content

    def ask_question(self, question):
        start_time = time.time()
        self.cache_stats['total_queries'] += 1
        cache_key = self.generate_cache_key(question)

        if cache_key in self.query_cache:
            self.cache_stats['cache_hits'] += 1
            cached_result = self.query_cache[cache_key]
            return cached_result

        self.cache_stats['cache_misses'] += 1

        retrieved_docs = self.retrieve_documents(question)
        answer = self.generate_answer(question, retrieved_docs)
        
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
