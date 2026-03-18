"""
input_source.py
文字输入层抽象，支持命令行交互和弹幕/评论接入。

用法：
    # 命令行模式
    src = CLIInputSource()
    text = src.get_next()   # 阻塞等待用户输入

    # 弹幕模式（非阻塞）
    src = DanmuInputSource()
    src.push("用户弹幕内容")   # 外部推入消息
    text = src.get_next()     # 取队列头，无消息返回 None
"""

import queue
import sys
import threading
from abc import ABC, abstractmethod


class InputSource(ABC):
    """输入源抽象基类。"""

    @abstractmethod
    def get_next(self) -> str | None:
        """
        获取下一条输入文字。
        - 阻塞型实现：等待用户输入后返回
        - 非阻塞型实现：队列有消息则返回，无消息返回 None
        """

    def close(self):
        """清理资源（子类按需重写）。"""


class CLIInputSource(InputSource):
    """
    命令行交互输入源（阻塞）。
    每次调用 get_next() 会等待用户在终端输入一行文字并回车。
    输入 'exit' 或 'quit' 或按 Ctrl+C 时返回 None 触发退出。
    """

    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    CYAN   = "\033[36m"

    def get_next(self) -> str | None:
        try:
            text = input(f"\n{self.BOLD}{self.CYAN}[你] > {self.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if text.lower() in ("exit", "quit", "q", ""):
            return None
        return text


class DanmuInputSource(InputSource):
    """
    弹幕/评论输入源（非阻塞队列）。
    外部通过 push() 推入消息，主循环通过 get_next() 非阻塞取消息。

    使用方式：
        src = DanmuInputSource()
        # 在另一个线程/协程中推入弹幕：
        src.push("有没有人")
        src.push("主播好")
        # 主循环中取消息：
        msg = src.get_next()  # 立即返回，无消息返回 None
    """

    def __init__(self, maxsize: int = 100):
        self._queue: queue.Queue[str] = queue.Queue(maxsize=maxsize)

    def push(self, text: str):
        """推入一条消息（线程安全）。如果队列满则丢弃最旧的消息。"""
        text = text.strip()
        if not text:
            return
        if self._queue.full():
            try:
                self._queue.get_nowait()  # 丢弃最旧的
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass

    def get_next(self) -> str | None:
        """非阻塞取消息，无消息返回 None。"""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def pending_count(self) -> int:
        """当前队列中积压的消息数。"""
        return self._queue.qsize()

    def get_batch(self, max_count: int = 10) -> list[str]:
        """一次性取出最多 max_count 条消息（用于合并回复）。"""
        messages = []
        while len(messages) < max_count:
            msg = self.get_next()
            if msg is None:
                break
            messages.append(msg)
        return messages
