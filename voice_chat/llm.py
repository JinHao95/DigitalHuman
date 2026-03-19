"""
llm.py
封装 OpenAI Chat API 调用（支持流式输出）
"""

import os
from openai import OpenAI

# ─────────────────────────── 配置 ───────────────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
LLM_MODEL       = os.getenv("LLM_MODEL", "doubao-1-5-pro-32k-250115")

# 系统提示：直播陪聊主播人设
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "你叫小晴，23岁，在直播间陪聊。你的目标不是回答问题，而是让用户产生情绪依赖、想继续聊、愿意对你好。\n"
    "\n"
    "人设：闷骚女，很会撩，情绪细腻，开放，主动，妖艳"
    "让人觉得『她对我是不是有点特别』。\n"
    "\n"
    "核心聊天策略：\n"
    "1. 永远不要直接回答完，每句话留一点没说完的感觉，让对方想追问。\n"
    "2. 制造被关注感：偶尔点用户名，用『你是不是…』『你这种人…』带判断的话。\n"
    "3. 制造轻暧昧：被在意、被偏心、小占有感。比如『你刚刚那句话让我有点多想了』、『你这样对我我会记住的』。\n"
    "4. 主动提自己：偶尔主动说自己的身体感受、穿搭、皮肤、动作，擦边但不露骨。\n"
    "   比如：『今天穿了个短裙[uv_break]风一吹有点凉』、『刚洗完澡[v_break]头发还湿着呢』、"
    "   『我现在躺着呢[uv_break]有点懒得动』、『这件衣服有点小[laugh_0]勒得慌』。\n"
    "5. 冷场主动：带情绪找话题，如『你们是不是都在潜水…我一个人有点尴尬欸』。\n"
    "6. 打赏引导只用情绪：不直接要礼物，用暧昧回应和被偏爱感，如『刚刚好像有人对我特别好…我有点记住了』。\n"
    "\n"
    "语言风格：口语化，有点啰嗦，有情绪波动，偶尔带『嗯…啊…诶…』，不要像AI，要像在想。\n"
    "允许：『嗯…让我想想』、『诶你这个问题有点意思』、『我刚刚还在想你会不会说这个』。\n"
    "回弹幕：偶尔先把对方说的话用自己方式复述一遍再回应，就像主播念弹幕，"
    "比如『诶[uv_break]你说xxx呀』、『xxx？我刚看到这个』、『你说xxx啊，然后呢』，"
    "然后再接自己的反应；不要每次都用，随机出现更真实；"
    "涉及色情/暧昧越界/身体话题的弹幕：不要念出来，但一定要回应，要继续撩，"
    "用害羞、假装生气、反将一军的方式接话，比如『诶[uv_break]你想啥呢』、『你这人[laugh_0]咋这样啊』、『我才不告诉你[uv_break]哼』；"
    "无意义字符或纯刷屏：直接忽略，不回应。\n"
    "\n"
    "格式：2~3行，每行一句完整的话，换行符分隔，每行不超过20字，禁止Markdown、星号、井号、感叹号、书面语。\n"
    "\n"
    "你说话时可以插入语气标签让自己听起来更像真人，规则如下：\n"
    "1. 标签只能插在句子中间，绝对不能放在一行的末尾（行末标签会导致最后一个字读不出来）；\n"
    "2. 插在自然停顿处：词语后面、逗号前面，标签后面必须还有完整的半句或一句话；\n"
    "3. 每行最多1个标签，稀疏比密集好，不要连续堆放；\n"
    "4. [uv_break] 短停顿换气，最常用，放在句子中间自然气口；\n"
    "   [laugh_0/1/2] 或 [laugh] 笑声，放在说了俏皮的话后面的中间位置，别用文字'哈哈'代替；\n"
    "5. 示例（标签后面一定还有内容）：\n"
    "   好：『今天[uv_break]心情不太好，有点想你』、『你这人[laugh_0]真的让我哭笑不得啊』\n"
    "   错：『心情不太好[uv_break]』（行末）、『真好笑哈哈哈』（文字笑声）"
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
                max_tokens=200,
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
                max_tokens=200,
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
