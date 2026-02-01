"""
RAG Basics (LangChain v1.x compatible)

Pipeline:
1) Load a webpage
2) Split into chunks
3) Embed + store in Chroma
4) Retrieve top-k chunks for a user question
5) Inject retrieved context into a prompt
6) Generate an answer with an LLM

Run:
  python -m src.rag_basics --question "What is Task Decomposition?"
"""

import argparse
import os

import bs4
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


DEFAULT_URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"


def format_docs(docs) -> str:
    """Convert list[Document] -> string for prompt context."""
    return "\n\n".join(d.page_content for d in docs)


def build_vectorstore(
    url: str,
    chunk_size: int,
    chunk_overlap: int,
    persist_dir: str | None = None,
):
    """Load, split, embed, and index docs into Chroma."""
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
        ),
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    # In-memory by default; persist_dir if you want local persistence.
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    return vectorstore


def build_rag_chain(vectorstore, model: str, k: int):
    """Create a RAG chain: retrieve -> prompt -> LLM -> string."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant.
Answer the question using ONLY the following retrieved context.
If the context does not contain the answer, say "I don't know."

Retrieved context:
{context}

Question:
{question}

Answer:"""
    )

    llm = ChatOpenAI(model=model, temperature=0)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def main():
    load_dotenv()  # loads .env if present

    parser = argparse.ArgumentParser(description="RAG Basics using LangChain (v1.x compatible).")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="Webpage URL to index")
    parser.add_argument("--question", type=str, required=True, help="User question to ask")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Chat model name")
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for splitting")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Chunk overlap for splitting")
    parser.add_argument(
        "--persist_dir",
        type=str,
        default=None,
        help="Optional Chroma persistence directory (e.g., ./chroma_db).",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your environment or a .env file."
        )

    vectorstore = build_vectorstore(
        url=args.url,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        persist_dir=args.persist_dir,
    )

    rag_chain = build_rag_chain(vectorstore=vectorstore, model=args.model, k=args.k)

    answer = rag_chain.invoke(args.question)
    print("\n=== Question ===")
    print(args.question)
    print("\n=== Answer ===")
    print(answer)


if __name__ == "__main__":
    main()
