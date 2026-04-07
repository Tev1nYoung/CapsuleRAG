from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List

from ..config import CapsuleRAGConfig
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Best-effort JSON object extraction.

    Some OpenAI-compatible servers/models do not strictly honor response_format,
    and will return:
    - fenced JSON
    - JSON with trailing commas
    - python-literal dicts (single quotes, True/False/None)
    - JSON missing a comma between top-level keys (common: nodes ... edges)
    """
    if not text:
        raise ValueError("empty response")

    raw = text.strip()
    candidates: List[str] = [raw]

    # Prefer the biggest fenced JSON block if present.
    if "```" in raw:
        blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
        if blocks:
            candidates.insert(0, max(blocks, key=len).strip())

    def _brace_substring(s: str) -> str:
        s = s.strip()
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end > start:
            return s[start : end + 1]
        return s

    candidates = [_brace_substring(c) for c in candidates]

    def _variants(s: str) -> List[str]:
        out = [s]
        # Remove trailing commas before } or ]
        out.append(re.sub(r",\s*([}\]])", r"\1", s))
        # Fix missing comma between top-level keys, e.g., `] "edges": ...`
        out.append(re.sub(r"(\])\s*(\"edges\"\s*:)", r"\1,\n\2", s))
        out.append(re.sub(r"(\})\s*(\"edges\"\s*:)", r"\1,\n\2", s))
        # Combine fixes
        s2 = re.sub(r",\s*([}\]])", r"\1", s)
        s2 = re.sub(r"(\])\s*(\"edges\"\s*:)", r"\1,\n\2", s2)
        s2 = re.sub(r"(\})\s*(\"edges\"\s*:)", r"\1,\n\2", s2)
        out.append(s2)
        # Dedup (preserve order)
        seen = set()
        uniq = []
        for x in out:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq

    last_err: Exception | None = None
    for c in candidates:
        for v in _variants(c):
            try:
                obj = json.loads(v)
                if isinstance(obj, dict):
                    return obj
            except Exception as e:
                last_err = e

            # Try Python-literal parse as a fallback.
            try:
                v_py = v
                v_py = re.sub(r"\bnull\b", "None", v_py, flags=re.IGNORECASE)
                v_py = re.sub(r"\btrue\b", "True", v_py, flags=re.IGNORECASE)
                v_py = re.sub(r"\bfalse\b", "False", v_py, flags=re.IGNORECASE)
                obj = ast.literal_eval(v_py)
                if isinstance(obj, dict):
                    return obj
            except Exception as e:
                last_err = e

    raise last_err or ValueError("failed to parse JSON object")


def _single_node_dag(question: str) -> Dict[str, Any]:
    return {
        "question": question,
        "nodes": [
            {"id": "q0", "question": question, "depends_on": [], "operator": "other", "vars_in": [], "vars_out": [], "constraints": {}}
        ],
        "edges": [],
    }


def _estimate_min_nodes(question: str, config: CapsuleRAGConfig) -> int:
    """
    Heuristic lower bound for decomposition depth.

    Musique-style hard questions often contain multiple nested relative clauses
    ("where ... who ... that ..."), and the LLM tends to collapse several hops
    into one node unless we ask for a finer-grained DAG.
    """
    q = str(question or "").lower()
    rel_hits = len(re.findall(r"\b(where|who|whose|which|that|when)\b", q))
    temporal_hits = len(re.findall(r"\b(after|before|during|while)\b", q))
    coord_hits = len(re.findall(r"\band\b", q))
    wh_hits = len(re.findall(r"\b(what|which|who|when|where|how)\b", q))

    target = 2 + min(rel_hits, 3)
    if temporal_hits:
        target += 1
    if coord_hits >= 2:
        target += 1
    if wh_hits >= 3 and rel_hits >= 2:
        target += 1
    return max(3, min(int(config.query_dag_max_nodes), target))


def _normalize_dag(question: str, dag: Dict[str, Any], config: CapsuleRAGConfig, meta: Any = None) -> Dict[str, Any]:
    nodes = dag.get("nodes") or []
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("DAG nodes missing")

    norm_nodes = []
    for idx, n in enumerate(nodes[: config.query_dag_max_nodes]):
        nid = n.get("id") or f"q{idx}"
        nq = n.get("question") or question
        norm_nodes.append(
            {
                "id": str(nid),
                "question": str(nq),
                "depends_on": list(n.get("depends_on") or []),
                "operator": str(n.get("operator") or "other"),
                "vars_in": list(n.get("vars_in") or []),
                "vars_out": list(n.get("vars_out") or []),
                "constraints": dict(n.get("constraints") or {}),
            }
        )

    edges = dag.get("edges") or []
    if not isinstance(edges, list):
        edges = []

    out = {"question": question, "nodes": norm_nodes, "edges": edges}
    if meta is not None:
        out["_llm_meta"] = meta
    return out


def _lines_to_chain_dag(question: str, lines: List[str], max_nodes: int) -> Dict[str, Any]:
    # Build a simple chain DAG q0 -> q1 -> q2 -> ...
    subqs = []
    for ln in lines:
        ln = str(ln or "").strip()
        if not ln:
            continue
        ln = re.sub(r"^\s*[\-\*\u2022]\s*", "", ln)  # bullets
        ln = re.sub(r"^\s*\d+[\)\.\:]\s*", "", ln)  # numbering
        ln = ln.strip().strip("\"").strip()
        if ln and ln.lower() != question.lower():
            subqs.append(ln)
    if not subqs:
        return _single_node_dag(question)

    subqs = subqs[: max(1, int(max_nodes))]
    nodes = []
    for i, sq in enumerate(subqs):
        nodes.append(
            {
                "id": f"q{i}",
                "question": sq,
                "depends_on": [] if i == 0 else [f"q{i-1}"],
                "operator": "other",
                "vars_in": [],
                "vars_out": [],
                "constraints": {},
            }
        )
    edges = [{"source": f"q{i-1}", "target": f"q{i}", "type": "depends"} for i in range(1, len(nodes))]
    return {"question": question, "nodes": nodes, "edges": edges}


def build_query_dag(question: str, llm, config: CapsuleRAGConfig) -> Dict[str, Any]:
    """
    Returns a JSON-serializable Query DAG dict.
    Falls back to a single-node DAG on parse/LLM failures.
    """
    if not config.enable_query_dag:
        return _single_node_dag(question)

    min_nodes = _estimate_min_nodes(question, config) if bool(getattr(config, "query_dag_enforce_min_nodes", False)) else 3
    system = (
        "You decompose a multi-hop question into a dependency DAG of retrieval-ready sub-questions. "
        "Your output will be used for dense retrieval embeddings, so ambiguity hurts performance. "
        "Output strict JSON only. Do not answer the question."
    )
    user = (
        "Question:\n"
        f"{question}\n\n"
        "Return JSON with keys: nodes, edges.\n"
        "Each node: {id, question, depends_on, operator, vars_in, vars_out, constraints}.\n"
        "Rules:\n"
        f"- Keep {min_nodes}-{config.query_dag_max_nodes} nodes if possible.\n"
        "- Prefer one atomic retrieval hop per node; do not merge multiple bridge steps into one node.\n"
        "- Every node.question MUST be a fully-specified, standalone question.\n"
        "- Do NOT use pronouns (he/she/it/they) and do NOT use placeholders like {location}, {symbol}, 'that place', 'the symbol'.\n"
        "- If needed, repeat entity names from the original question so each node.question is directly searchable.\n"
        "- Keep vars_in and vars_out as empty lists (we are not executing variable binding).\n"
        "- operator in [bridge,comparison,intersection,temporal,attribute,other].\n"
        "- constraints may include given_entities, answer_type, negation.\n"
        "- Output JSON only."
    )
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    def _llm_json(messages_: List[Dict[str, str]]):
        try:
            return llm.infer(messages=messages_, response_format={"type": "json_object"})
        except Exception:
            # Some OpenAI-compatible servers may not support response_format.
            return llm.infer(messages=messages_)

    try:
        raw, meta = _llm_json(messages)
        dag = _extract_json_object(raw)
        out = _normalize_dag(question=question, dag=dag, config=config, meta=meta)
        if len(out.get("nodes") or []) >= min_nodes or min_nodes <= 3:
            return out
        raise ValueError(f"under_decomposed:{len(out.get('nodes') or [])}<{min_nodes}")
    except Exception as e:
        # Second attempt: if the first DAG is too shallow or malformed, force a finer-grained expansion.
        try:
            system2 = (
                "Return a valid JSON object only. Do not include any extra text. "
                "Decompose the question into finer-grained retrieval hops."
            )
            user2 = (
                f"Question:\n{question}\n\n"
                "Return JSON with keys: nodes, edges.\n"
                "nodes is a list of objects {id, question, depends_on, operator, vars_in, vars_out, constraints}.\n"
                "edges is a list.\n"
                f"Use at least {min_nodes} nodes and at most {config.query_dag_max_nodes} nodes if possible.\n"
                "Do not merge multiple bridge relations into one node.\n"
                "Each node.question should resolve exactly one retrieval step and remain standalone.\n"
                "Use double quotes for all strings.\n"
                "Do not output markdown."
            )
            raw2, meta2 = _llm_json([{"role": "system", "content": system2}, {"role": "user", "content": user2}])
            dag2 = _extract_json_object(raw2)
            out2 = _normalize_dag(question=question, dag=dag2, config=config, meta=meta2)
            if len(out2.get("nodes") or []) >= max(2, min_nodes - 1):
                return out2
        except Exception:
            pass

        # Third attempt: line-based decomposition (more robust than JSON for some providers).
        try:
            system3 = (
                "Decompose the question into short, standalone sub-questions. "
                "Output one sub-question per line. No numbering, no bullets, no extra text. "
                "Do not merge multiple bridge hops into one line."
            )
            user3 = f"Question:\n{question}\n\nReturn between {min_nodes} and {config.query_dag_max_nodes} lines."
            raw3, _meta3 = llm.infer(messages=[{"role": "system", "content": system3}, {"role": "user", "content": user3}])
            lines = [ln.strip() for ln in str(raw3 or "").splitlines() if ln.strip()]
            if len(lines) >= max(2, min_nodes - 1):
                return _lines_to_chain_dag(question, lines, config.query_dag_max_nodes)
        except Exception:
            pass

        logger.warning(f"[CapsuleRAG] [ONLINE_QDAG] fallback to single-node DAG | err={type(e).__name__}: {e}")
        return _single_node_dag(question)
