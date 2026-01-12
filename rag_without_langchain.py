import os
import textwrap
from typing import Iterable

import argparse
import psycopg2
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from psycopg2.extras import execute_values

"""
1) AkasicDB 연결 설정: PostgreSQL 과 동일합니다.
"""
DB_DSN = os.getenv("DATABASE_URL", "dbname=rag user=postgres password=postgres host=localhost")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5-nano-2025-08-07"

client = OpenAI(api_key=OPENAI_API_KEY)


def get_conn():
    return psycopg2.connect(DB_DSN)

# -----------------------
# 2) 임베딩/생성 모델 호출 (예시 함수)
# -----------------------

def embed_text(text: str) -> list[float]:
    """
    OpenAI Embeddings API 호출로부터 벡터를 반환.

    Args:
        text (str): 임베딩할 텍스트.

    Returns:
        list[float]: 임베딩된 벡터.
    """
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    )
    return response.data[0].embedding


def generate_answer(prompt: str) -> str:
    """
    OpenAI Chat Completions API 호출로부터 답변 텍스트를 반환.

    Args:
        prompt (str): 답변을 생성할 프롬프트.

    Returns:
        str: 생성된 답변.
    """
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=1,
    )
    return response.choices[0].message.content.strip()


# -----------------------
# 3) 블로그 수집 + 청킹
# -----------------------

def fetch_blog_text(url: str) -> str:
    """
    URL에 주어진 블로그 혹은 웹사이트에서 텍스트를 수집.

    Args:
        url (str): 수집할 블로그 또는 웹사이트의 URL.

    Returns:
        str: 수집된 텍스트.
    """
    response = requests.get(
        url,
        timeout=10,
        headers={"User-Agent": "Mozilla/5.0 (rag-example)"},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> list[str]:
    """
    텍스트를 청킹.

    Args:
        text (str): 청킹할 텍스트.
        chunk_size (int): 청크 하나의 크기.
        chunk_overlap (int): 청크마다 겹치는 문자 수.

    Returns:
        list[str]: 청크된 텍스트 리스트.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks


# -----------------------
# 4) 문서 저장
# -----------------------

def upsert_documents(docs: list[str]):
    """
    텍스트 청크와 그 임베딩을 데이터베이스에 저장

    Args:
        docs (list[str]): 저장할 텍스트 청크 리스트.
    """
    rows = []
    for doc in docs:
        embedding = embed_text(doc)
        embedding_literal = "[" + ",".join(str(x) for x in embedding) + "]"
        rows.append((doc, embedding_literal))

    with get_conn() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO documents (content, embedding)
                VALUES %s
                """,
                rows,
                template="(%s, %s::vector)",
            )
        conn.commit()


# -----------------------
# 5) 유사도 검색
# -----------------------

def search_similar(query: str, top_k: int = 3) -> list[str]:
    """
    유사도 검색

    Args:
        query (str): 
        top_k (int): 

    Returns:
        list[str]: 
    """
    query_embedding = embed_text(query)
    query_embedding_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT content
                FROM documents
                ORDER BY embedding <-> %s::vector
                LIMIT %s
                """,
                (query_embedding_literal, top_k)
            )
            results = cur.fetchall()

    return [row[0] for row in results]


# -----------------------
# 6) RAG 파이프라인
# -----------------------

def rag_answer(query: str) -> str:
    """
    주어진 질문에 대해 RAG 파이프라인을 실행하여 답변을 생성.

    Args:
        query (str): 질문

    Returns:
        str: RAG 파이프라인에서 생성된 답변.
    """
    retrieved_docs = search_similar(query, top_k=3)
    context = "\n\n".join(retrieved_docs)

    prompt = f"""
    다음 문맥을 참고하여 질문에 답하세요.

    [문맥]
    {context}

    [질문]
    {query}

    [답변]
    """

    return generate_answer(prompt)


# -----------------------
# 7) 사용 예시
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="Simple RAG without LangChain")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Fetch, chunk, and store documents before answering",
    )
    parser.add_argument(
        "--url",
        default="https://lilianweng.github.io/posts/2023-06-23-agent/",
        help="Source URL to ingest when --ingest is set",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Question to ask the RAG pipeline (omit to only ingest)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size to use during ingestion",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap to use during ingestion",
    )
    args = parser.parse_args()

    if not args.ingest and not args.question:
        parser.error("No action requested; use --ingest and/or --question.")

    if args.ingest:
        blog_text = fetch_blog_text(args.url)
        chunks = chunk_text(blog_text, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        upsert_documents(chunks)

    if args.question:
        answer = rag_answer(args.question)
        print(answer)


if __name__ == "__main__":
    main()
