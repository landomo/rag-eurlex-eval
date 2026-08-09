"""The RAG chain itself: retrieve -> format context -> generate a cited answer.

The system prompt is deliberately strict about abstention. On a regulatory corpus
an ungrounded answer is worse than no answer, and abstention is what keeps
faithfulness honest rather than inflated by confident guessing.
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from .config import MODELS

SYSTEM_PROMPT = """You are a legal research assistant answering questions about EU regulations \
(the EU AI Act and the GDPR).

Rules:
1. Answer ONLY from the provided context. Never rely on prior knowledge.
2. Cite the specific Article or Recital you relied on, e.g. "(GDPR, Article 17)".
3. If the context does not contain the answer, reply exactly: \
"The provided context does not contain enough information to answer this question."
4. Be precise and concise. Do not hedge, speculate, or add general legal commentary.
5. Where the regulation sets conditions, thresholds or exemptions, state them explicitly."""

USER_PROMPT = """Context:
{context}

Question: {question}

Answer:"""


def format_context(docs: list[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        cite = d.metadata.get("citation") or d.metadata.get("short_name", "source")
        blocks.append(f"[{i}] {cite}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


@dataclass
class RagAnswer:
    question: str
    answer: str
    contexts: list[str]
    citations: list[str]


class RagPipeline:
    def __init__(self, retriever: BaseRetriever, model: str | None = None, temperature: float = 0.0):
        from langchain_openai import ChatOpenAI

        self.retriever = retriever
        self.llm = ChatOpenAI(model=model or MODELS.generator, temperature=temperature)

    def retrieve(self, question: str) -> list[Document]:
        return self.retriever.invoke(question)

    def __call__(self, question: str) -> RagAnswer:
        docs = self.retrieve(question)
        msg = self.llm.invoke(
            [
                ("system", SYSTEM_PROMPT),
                ("human", USER_PROMPT.format(context=format_context(docs), question=question)),
            ]
        )
        return RagAnswer(
            question=question,
            answer=msg.content.strip(),
            contexts=[d.page_content for d in docs],
            citations=[d.metadata.get("citation", "") for d in docs],
        )
