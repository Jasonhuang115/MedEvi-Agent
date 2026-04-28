"""
MedEvi-Agent LangGraph workflow

Pipeline:
  search ──▶ screen ──▶ extract ──▶ synthesis ──▶ END
    │            │
    │            ▼ (no papers)
    │           END
    │
    ▼ (0 results)
  synthesis ──▶ END
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from agents import extract_agent, screen_agent, search_agent, synthesis_agent
from agents.common import state_get
from state import PaperState

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════
# Conditional routing
# ═══════════════════════════════════════════

def _route_after_search(state: Any) -> str:
    """Skip downstream nodes if search returned nothing."""
    pubmed_ids = state_get(state, "pubmed_ids", []) or []
    reranked = state_get(state, "reranked_abstracts", []) or []
    if not pubmed_ids and not reranked:
        logger.info("Search returned 0 results — skipping to synthesis")
        return "synthesis"
    return "screen"


def _route_after_screen(state: Any) -> str:
    """Only proceed to extraction if papers passed screening."""
    if hasattr(state, "screened_papers"):
        screened = state.screened_papers or []
    else:
        screened = state.get("screened_papers", []) or []
    return "extract" if screened else END


# ═══════════════════════════════════════════
# Node wrapper: catch unhandled exceptions
# ═══════════════════════════════════════════

def _safe_node(fn, name: str):
    """Wrap an agent node so unexpected exceptions don't crash the pipeline."""
    def wrapper(state: Any) -> Dict[str, Any]:
        try:
            return fn(state)
        except Exception as exc:
            logger.exception("Node '%s' failed: %s", name, exc)
            return {"error": f"[{name}] {exc}"}
    return wrapper


# ═══════════════════════════════════════════
# Build graph
# ═══════════════════════════════════════════

def build_app():
    workflow = StateGraph(PaperState)

    workflow.add_node("search", _safe_node(search_agent, "search"))
    workflow.add_node("screen", _safe_node(screen_agent, "screen"))
    workflow.add_node("extract", _safe_node(extract_agent, "extract"))
    workflow.add_node("synthesis", _safe_node(synthesis_agent, "synthesis"))

    workflow.set_entry_point("search")
    workflow.add_conditional_edges("search", _route_after_search, {
        "screen": "screen",
        "synthesis": "synthesis",
    })
    workflow.add_conditional_edges("screen", _route_after_screen)
    workflow.add_edge("extract", "synthesis")
    workflow.add_edge("synthesis", END)

    return workflow.compile()


app = build_app()
