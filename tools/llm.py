"""
DeepSeek LLM Wrapper
使用OpenAI兼容接口调用DeepSeek API
"""
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
