import re

_CYRILLIC = re.compile(r"[а-яёА-ЯЁ]")
_CJK = re.compile(r"[぀-ヿ㐀-鿿＀-￯。]")


def detect_language(text: str) -> str:
    """Language the answer must be written in: Russian if the question has Cyrillic, else English."""
    return "Russian" if _CYRILLIC.search(text) else "English"


def has_cjk(text: str) -> bool:
    """Qwen models sometimes slip into Chinese mid-answer."""
    return bool(_CJK.search(text))
