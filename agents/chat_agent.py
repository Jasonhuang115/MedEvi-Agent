"""Chat Agent: ReAct 循环 + Tool Use。

在四节点管线之外的第 5 个 Agent。
接收管线结果作为数据上下文，用户可自由追问分析结果。
"""

from agents.llm_router import is_api_available
from agents.chat_tools import CHAT_TOOLS, build_handlers
from tools.llm import DeepSeekLLM

CHAT_SYSTEM_PROMPT = """你是一个循证医学研究助手。用户已经用 MedEvi-Agent 完成了一次文献检索与分析（PubMed 检索、PICOS筛选、数据提取和GRADE证据评级），你可以通过工具函数访问此次分析的全部数据。

回答规则：
1. 所有数据必须来自工具调用返回的结果，不要编造
2. 工具返回的数据是 Python 计算的，可以信任
3. 看到效应量、CI 等数值时，必须从工具结果中引用，不要自己计算
4. 用中文回答，医学名词保留英文原文
5. 回答要简洁，不需要重复整个报告"""


def chat_response(
    user_message: str,
    pipeline_context: dict,
    chat_history: list[dict] = None,
) -> tuple[str, list[dict]]:
    """处理用户的一条消息，返回 Agent 回复。

    Args:
        user_message: 用户输入
        pipeline_context: 管线运行结果 {
            'extracted_picos': [...],
            'quantitative_outcomes': [...],
            'screened_papers': [...],
            'grade_report': str,
            'query': str,
        }
        chat_history: 之前的对话 [{"user": ..., "assistant": ..., "tool_logs": ...}]

    Returns:
        (reply_text, tool_logs)
    """
    if not is_api_available():
        return ("DeepSeek API 不可用，对话功能暂时无法使用。请检查 API Key 配置。", [])

    tool_handlers = build_handlers(pipeline_context)

    # 构建消息
    messages = []
    if chat_history:
        for turn in chat_history[-6:]:
            messages.append({"role": "user", "content": turn["user"]})
            if turn.get("assistant"):
                messages.append(
                    {"role": "assistant", "content": turn["assistant"]}
                )

    messages.append({"role": "user", "content": user_message})

    llm = DeepSeekLLM(model="deepseek-chat", temperature=0.3, max_tokens=2048)
    return llm.invoke_with_tools(
        CHAT_SYSTEM_PROMPT, messages, CHAT_TOOLS, tool_handlers
    )
