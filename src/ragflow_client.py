"""RAGFlow HTTP 客户端（仅用 requests 标准库）。

端点依据 RAGFlow 官方 HTTP API（v0.15+，2025-2026）：
- POST /api/v1/retrieval                       检索 chunks
- POST /api/v1/openai/{chat_id}/chat/completions  问答（OpenAI 兼容，非流式）
- GET  /api/v1/datasets/{id}/documents         文档/元数据核查（DATA-02/03 用）
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests

from .models import QAOutcome, RetrievedChunk


class RAGFlowClient:
    def __init__(self, base_url: str, api_key: str, timeout_s: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_s
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        })

    # ---------- 基础 ----------
    def health(self) -> Dict[str, Any]:
        """健康检查：列出数据集（鉴权+连通二合一）。"""
        r = self.session.get(f"{self.base_url}/api/v1/datasets?page=1&page_size=1",
                             timeout=self.timeout)
        return {"http_status": r.status_code, "ok": r.status_code < 400,
                "body": r.text[:500]}

    # ---------- 检索 ----------
    def retrieve(self, question: str, dataset_ids: List[str],
                 page: int = 1, page_size: int = 30, top_k: int = 1024,
                 similarity_threshold: float = 0.2,
                 vector_similarity_weight: float = 0.3,
                 keyword: bool = False, highlight: bool = False,
                 rerank_id: Optional[str] = None,
                 document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """返回 {'chunks': [RetrievedChunk], 'latency_ms': int, 'raw': dict}。"""
        body: Dict[str, Any] = {
            "question": question,
            "dataset_ids": dataset_ids,
            "page": page,
            "page_size": page_size,
            "similarity_threshold": similarity_threshold,
            "vector_similarity_weight": vector_similarity_weight,
            "top_k": top_k,
            "keyword": keyword,
            "highlight": highlight,
        }
        if rerank_id:
            body["rerank_id"] = rerank_id
        if document_ids:
            body["document_ids"] = document_ids
        t0 = time.perf_counter()
        r = self.session.post(f"{self.base_url}/api/v1/retrieval",
                              json=body, timeout=self.timeout)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        r.raise_for_status()
        payload = r.json()
        chunks = self._parse_chunks(payload, page_size)
        return {"chunks": chunks, "latency_ms": latency_ms, "raw": payload}

    @staticmethod
    def _parse_chunks(payload: Dict[str, Any], page_size: int) -> List[RetrievedChunk]:
        data = payload.get("data", payload)
        raw_list = data.get("chunks") or data.get("chunk_list") or data.get("ret") or []
        out: List[RetrievedChunk] = []
        for i, c in enumerate(raw_list[:page_size]):
            if not isinstance(c, dict):
                continue
            out.append(RetrievedChunk(
                chunk_id=str(c.get("chunk_id") or c.get("id") or ""),
                document_id=str(c.get("document_id") or c.get("doc_id") or ""),
                document_name=str(c.get("document_name") or c.get("docnm_kwd") or ""),
                content=str(c.get("content") or c.get("content_with_weight") or ""),
                score=float(c.get("similarity") or c.get("score") or 0.0),
                rank=i + 1,
                metadata={k: v for k, v in c.items()
                          if k not in ("content", "content_with_weight")},
            ))
        return out

    # ---------- 问答 ----------
    def chat(self, chat_id: str, question: str,
             with_reference: bool = True) -> QAOutcome:
        """OpenAI 兼容接口，非流式，一次问答。reference 供 RAG-06/GEN-10 分析。"""
        url = f"{self.base_url}/api/v1/openai/{chat_id}/chat/completions"
        body = {
            "model": "model",   # 占位：沿用 Chat Assistant 配置的模型
            "messages": [{"role": "user", "content": question}],
            "stream": False,
            "extra_body": {"reference": with_reference},
        }
        t0 = time.perf_counter()
        try:
            r = self.session.post(url, json=body, timeout=self.timeout)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            r.raise_for_status()
            data = r.json()
            msg = data["choices"][0]["message"]
            refs = msg.get("reference") or {}
            if isinstance(refs, dict):
                ref_list = refs.get("chunks") or []
            elif isinstance(refs, list):
                ref_list = refs
            else:
                ref_list = []
            return QAOutcome(question_id="", answer=str(msg.get("content") or ""),
                             references=ref_list if isinstance(ref_list, list) else [],
                             generation_latency_ms=latency_ms,
                             e2e_latency_ms=latency_ms, trace_ok=True)
        except Exception as exc:  # noqa: BLE001 — 测试客户端如实记录错误
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return QAOutcome(question_id="", answer="",
                             generation_latency_ms=latency_ms,
                             e2e_latency_ms=latency_ms, trace_ok=False,
                             error=str(exc))

    # ---------- 文档/元数据 ----------
    def list_documents(self, dataset_id: str, page: int = 1,
                       page_size: int = 100) -> Dict[str, Any]:
        r = self.session.get(
            f"{self.base_url}/api/v1/datasets/{dataset_id}/documents",
            params={"page": page, "page_size": page_size}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_document(self, dataset_id: str, document_id: str) -> Dict[str, Any]:
        r = self.session.get(
            f"{self.base_url}/api/v1/datasets/{dataset_id}/documents/{document_id}",
            timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def list_chunks(self, dataset_id: str, document_id: str, page: int = 1,
                    page_size: int = 100, keywords: str = "") -> Dict[str, Any]:
        params: Dict[str, Any] = {"page": page, "page_size": page_size}
        if keywords:
            params["keywords"] = keywords
        r = self.session.get(
            f"{self.base_url}/api/v1/datasets/{dataset_id}/documents/"
            f"{document_id}/chunks",
            params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def list_all_chunks(self, dataset_id: str, document_id: str,
                        page_size: int = 100) -> List[RetrievedChunk]:
        out: List[RetrievedChunk] = []
        page = 1
        while True:
            payload = self.list_chunks(dataset_id, document_id, page, page_size)
            data = payload.get("data", payload)
            raw = data.get("chunks") or data.get("chunk_list") or []
            if not raw:
                break
            for c in raw:
                if not isinstance(c, dict):
                    continue
                out.append(RetrievedChunk(
                    chunk_id=str(c.get("chunk_id") or c.get("id") or ""),
                    document_id=str(c.get("document_id") or c.get("doc_id")
                                    or document_id),
                    document_name=str(c.get("document_name") or c.get("docnm_kwd") or ""),
                    content=str(c.get("content") or ""),
                    score=0.0,
                    rank=len(out) + 1,
                    metadata={k: v for k, v in c.items() if k != "content"},
                ))
            total = data.get("total")
            if total is not None and len(out) >= int(total):
                break
            if len(raw) < page_size:
                break
            page += 1
        return out
