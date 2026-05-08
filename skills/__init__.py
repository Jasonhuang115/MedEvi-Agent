"""Skills 层：独立的、可复用的医学文献分析能力函数。

每个 skill 是纯函数，不依赖 LangGraph state。
所有 skill 调用统一经 _safe_call() 包装异常边界。
"""


def _safe_call(fn, *args, **kwargs):
    """统一异常保护：skill 出错时返回 error dict，不抛异常。

    Chat Agent 和 MCP Server 统一使用此函数调用 skill，
    各自不再需要重复的 try/except。
    """
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return {"error": f"{fn.__name__} failed: {e}"}


# 延迟导入，避免循环依赖
def search(snp, disease, max_results=50):
    from skills.skill_search import search as _search

    return _safe_call(_search, snp, disease, max_results)


def screen_single(title, abstract, pico_query, query):
    from skills.skill_screen import screen_single as _ss

    return _safe_call(_ss, title, abstract, pico_query, query)


def extract_picos(abstract):
    from skills.skill_extract import extract_picos as _ep

    return _safe_call(_ep, abstract)


def extract_numerical(abstract):
    from skills.skill_extract import extract_numerical as _en

    return _safe_call(_en, abstract)


def compute_stats(extracted, quantitative, screened):
    from skills.skill_stats import compute_stats as _cs

    return _safe_call(_cs, extracted, quantitative, screened)


def retrieve_guidelines(topic, top_k=4):
    from skills.skill_guideline import retrieve_guidelines as _rg

    return _safe_call(_rg, topic, top_k)
