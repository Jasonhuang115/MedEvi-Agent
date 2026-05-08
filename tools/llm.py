"""
DeepSeek LLM Wrapper
使用OpenAI兼容接口调用DeepSeek API，支持 Function Calling (Tool Use)
"""
import json
import os
from openai import OpenAI


class DeepSeekLLM:
    """DeepSeek模型包装器（OpenAI兼容接口）"""

    def __init__(
        self,
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, prompt: str, temperature: float | None = None) -> str:
        """调用DeepSeek API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"DeepSeek API调用失败: {e}")
            return ""

    def invoke_with_tools(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
        tool_handlers: dict,
        max_rounds: int = 8,
    ) -> tuple[str, list[dict]]:
        """ReAct 循环：LLM 决策 → 执行工具 → 观察 → 继续，直到 LLM 给出最终回答。

        Args:
            system_prompt: 系统提示词
            messages: 对话消息列表 [{"role": "user/assistant", "content": ...}]
            tools: OpenAI function calling 格式的工具 schema
            tool_handlers: {"function_name": callable} 映射
            max_rounds: 最大 ReAct 循环轮次

        Returns:
            (final_response, tool_logs)
            - final_response: LLM 的最终文本回复
            - tool_logs: [{round, tool, args, result_summary}]
        """
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        tool_logs = []

        for _ in range(max_rounds):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=full_messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            except Exception as e:
                print(f"invoke_with_tools API调用失败: {e}")
                return (f"API 调用失败: {e}", tool_logs)

            msg = response.choices[0].message

            if msg.tool_calls:
                full_messages.append(msg)
                for tc in msg.tool_calls:
                    func_name = tc.function.name
                    try:
                        func_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        func_args = {}

                    handler = tool_handlers.get(func_name)
                    try:
                        result = (
                            handler(**func_args)
                            if handler
                            else {"error": f"未知工具: {func_name}"}
                        )
                    except Exception as e:
                        result = {"error": f"工具 {func_name} 执行失败: {e}"}

                    tool_logs.append({
                        "round": len(tool_logs) + 1,
                        "tool": func_name,
                        "args": func_args,
                        "result_summary": str(result)[:300],
                    })

                    full_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(
                            result, ensure_ascii=False, default=str
                        ),
                    })
            else:
                return (msg.content or "抱歉，我无法回答这个问题。", tool_logs)

        return ("处理超时，请尝试简化您的问题。", tool_logs)


# 全局实例缓存
_llm_instance = None


def get_llm(model: str = "deepseek-chat", temperature: float = 0.7) -> DeepSeekLLM:
    """
    获取DeepSeek模型实例

    Args:
        model: 模型名称
            - deepseek-chat: DeepSeek-V3 对话模型
        temperature: 温度参数 0-2
    """
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = DeepSeekLLM(model=model, temperature=temperature)
    return _llm_instance
