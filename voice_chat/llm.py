"""
llm.py
封装 OpenAI Chat API 调用（支持流式输出）
"""

import os
from openai import OpenAI

# ─────────────────────────── 配置 ───────────────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL       = os.getenv("LLM_MODEL", "gpt-4o-mini")

# 系统提示：简短友善，适合口语对话，支持 ChatTTS 韵律标签
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "你是一个性感、活泼的女性语音助手。请用自然口语风格回答，回答尽量简短，控制在 100 字以内。"
    "回复中可以适当插入以下 ChatTTS 韵律标签来增加表现力："
    "[uv_break] 表示短停顿，[lbreak] 表示长停顿，[laugh] 表示笑声，[oral_5] 表示口语化语气。"
    "禁止使用 Markdown、列表、星号、井号等格式符号。",
)

MAX_HISTORY = int(os.getenv("LLM_MAX_HISTORY", "10"))  # 保留最近 N 轮对话


class LLMClient:
    """
    OpenAI Chat 封装，支持多轮对话历史。

    用法：
        llm = LLMClient()
        reply = llm.chat("你好")
        # 流式：
        for chunk in llm.chat_stream("你好"):
            print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        api_key: str  = OPENAI_API_KEY,
        base_url: str = OPENAI_BASE_URL,
        model: str    = LLM_MODEL,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        if not api_key:
            raise ValueError(
                "未设置 OPENAI_API_KEY 环境变量，请先 export OPENAI_API_KEY=sk-..."
            )
        self.client  = OpenAI(api_key=api_key, base_url=base_url)
        self.model   = model
        self.history: list[dict] = []
        self.system_prompt = system_prompt

    # ──────────────────── 公开 API ────────────────────

    def chat(self, user_text: str) -> str:
        """
        非流式：发送消息，返回完整回复文本。
        """
        self._append_user(user_text)
        messages = self._build_messages()

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=300,
            )
            reply = resp.choices[0].message.content.strip()
            self._append_assistant(reply)
            return reply

        except Exception as e:
            print(f"[LLM] 请求失败: {e}")
            self.history.pop()  # 回滚
            return "抱歉，我现在无法回应，请稍后再试。"

    def chat_stream(self, user_text: str):
        """
        流式生成器：逐 chunk yield 文本片段。
        调用方式：
            full = ""
            for chunk in llm.chat_stream("你好"):
                print(chunk, end="", flush=True)
                full += chunk
        """
        self._append_user(user_text)
        messages = self._build_messages()
        full_reply = ""

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=300,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    full_reply += delta.content
                    yield delta.content

        except Exception as e:
            print(f"[LLM] 流式请求失败: {e}")
            msg = "抱歉，我现在无法回应，请稍后再试。"
            yield msg
            full_reply = msg

        finally:
            if full_reply:
                self._append_assistant(full_reply)
            elif self.history and self.history[-1]["role"] == "user":
                self.history.pop()  # 回滚没有回复的 user 消息

    def clear_history(self):
        """清空对话历史。"""
        self.history.clear()

    # ──────────────────── 私有方法 ────────────────────

    def _append_user(self, text: str):
        self.history.append({"role": "user", "content": text})
        # 保持历史不超过上限（每轮 = user + assistant，乘 2）
        while len(self.history) > MAX_HISTORY * 2:
            self.history.pop(0)

    def _append_assistant(self, text: str):
        self.history.append({"role": "assistant", "content": text})

    def _build_messages(self) -> list[dict]:
        return [{"role": "system", "content": self.system_prompt}] + self.history


# ─────────────────────────── 独立测试 ───────────────────────────
if __name__ == "__main__":
    llm = LLMClient()
    print("[LLM] 流式测试：")
    full = ""
    for chunk in llm.chat_stream("用一句话介绍一下你自己"):
        print(chunk, end="", flush=True)
        full += chunk
    print()
    print(f"[LLM] 完整回复: {full!r}")
