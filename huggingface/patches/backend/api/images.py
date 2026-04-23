"""
图片生成接口补丁版 — 兼容 OpenAI /v1/images/generations 规范。

相对上游版本的增强点：
1. 在原始正则提取外，递归扫描 SSE 事件和 chat 详情中的嵌套 URL 字段。
2. 对常见图片字段名做宽松匹配，减少上游返回结构变化导致的 no URL found。
3. 在失败时输出结构摘要日志，便于 HF 线上排查。
"""

import asyncio
import base64
import json
import logging
import re
import time
from collections.abc import Iterable
from urllib.request import Request as UrlRequest, urlopen

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from backend.services.qwen_client import QwenClient

log = logging.getLogger("qwen2api.images")
router = APIRouter()

DEFAULT_IMAGE_MODEL = "qwen3.6-plus"

IMAGE_MODEL_MAP = {
    "dall-e-3": "qwen3.6-plus",
    "dall-e-2": "qwen3.6-plus",
    "qwen-image": "qwen3.6-plus",
    "qwen-image-plus": "qwen3.6-plus",
    "qwen-image-turbo": "qwen3.6-plus",
    "qwen3.6-plus": "qwen3.6-plus",
}

_URL_KEY_RE = re.compile(r"(?:^|_)(?:url|image|src)$|imageurl|image_url|imageuri|image_uri", re.IGNORECASE)
_CDN_URL_RE = re.compile(
    r'https?://(?:cdn\.qwenlm\.ai|wanx\.alicdn\.com|img\.alicdn\.com|[^\s"<>]+\.(?:jpg|jpeg|png|webp|gif))(?:[^\s"<>]*)',
    re.IGNORECASE,
)


def _append_unique(target: list[str], value: str) -> None:
    if value and value not in target:
        target.append(value)


def _extract_image_urls_from_text(text: str) -> list[str]:
    urls: list[str] = []

    for u in re.findall(r'!\[.*?\]\((https?://[^\s\)]+)\)', text):
        _append_unique(urls, u.rstrip(").,;"))

    for u in re.findall(r'"(?:url|image|src|imageUrl|image_url)"\s*:\s*"(https?://[^"]+)"', text):
        _append_unique(urls, u)

    for u in _CDN_URL_RE.findall(text):
        _append_unique(urls, u.rstrip(".,;)\"'>"))

    return urls


def _extract_image_urls_from_node(node) -> list[str]:
    urls: list[str] = []

    def walk(value, parent_key: str = "") -> None:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("http://") or stripped.startswith("https://"):
                if _URL_KEY_RE.search(parent_key) or _CDN_URL_RE.search(stripped):
                    _append_unique(urls, stripped)
            for match in _extract_image_urls_from_text(value):
                _append_unique(urls, match)
            return

        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, str(key or ""))
            return

        if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
            for child in value:
                walk(child, parent_key)

    walk(node)
    return urls


def _extract_image_urls(text: str, payloads: list[dict] | None = None, current_chat: dict | None = None) -> list[str]:
    urls = _extract_image_urls_from_text(text)
    for payload in payloads or []:
        for url in _extract_image_urls_from_node(payload):
            _append_unique(urls, url)
    if current_chat:
        for url in _extract_image_urls_from_node(current_chat):
            _append_unique(urls, url)
    return urls


def _resolve_image_model(requested: str | None) -> str:
    if not requested:
        return DEFAULT_IMAGE_MODEL
    return IMAGE_MODEL_MAP.get(requested, DEFAULT_IMAGE_MODEL)


def _get_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key", "").strip()


def _download_image_as_base64(url: str) -> str:
    request = UrlRequest(url, method="GET")
    with urlopen(request, timeout=120) as response:
        return base64.b64encode(response.read()).decode("ascii")


def _build_image_prompt(prompt: str) -> str:
    return (
        "请直接生成图片，不要只输出文字描述。"
        "如果可以生成图片，请返回可访问的图片链接或包含图片链接的结果。\n\n"
        f"用户需求：{prompt}"
    )


def _summarize_payload(payload) -> str:
    try:
        serialized = json.dumps(payload, ensure_ascii=False)
    except Exception:
        serialized = str(payload)
    return serialized[:1000]


@router.post("/v1/images/generations")
@router.post("/images/generations")
async def create_image(request: Request):
    from backend.core.config import API_KEYS, settings

    client: QwenClient = request.app.state.qwen_client

    token = _get_token(request)
    if API_KEYS:
        if token != settings.ADMIN_KEY and token not in API_KEYS:
            raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    prompt: str = body.get("prompt", "").strip()
    if not prompt:
        raise HTTPException(400, "prompt is required")

    n: int = min(max(int(body.get("n", 1)), 1), 4)
    model = _resolve_image_model(body.get("model"))
    response_format = str(body.get("response_format", "url") or "url").strip().lower()

    log.info(f"[T2I] model={model}, n={n}, prompt={prompt[:80]!r}")

    acc = None
    chat_id = None
    current_chat = None
    try:
        prompt_text = _build_image_prompt(prompt)
        event_payloads: list[dict] = []
        event_payload_texts: list[str] = []

        async for item in client.chat_stream_events_with_retry(model, prompt_text, has_custom_tools=False):
            if item.get("type") == "meta":
                acc = item.get("acc")
                chat_id = item.get("chat_id")
                continue
            if item.get("type") != "event":
                continue
            evt = item.get("event", {})
            event_payloads.append(evt)
            event_payload_texts.append(json.dumps(evt, ensure_ascii=False))

        if acc is None or chat_id is None:
            raise HTTPException(status_code=500, detail="Image generation session was not created")

        chats = await client.list_chats(acc.token, limit=20)
        current_chat = next((c for c in chats if isinstance(c, dict) and c.get("id") == chat_id), None)
        answer_text = "\n".join(event_payload_texts)
        if current_chat:
            answer_text += "\n" + json.dumps(current_chat, ensure_ascii=False)

        image_urls = _extract_image_urls(answer_text, payloads=event_payloads, current_chat=current_chat)
        log.info(f"[T2I] 提取到 {len(image_urls)} 张图片 URL: {image_urls}")

        if not image_urls:
            log.warning("[T2I] 未提取到图片 URL，event 摘要: %s", _summarize_payload(event_payloads))
            log.warning("[T2I] 未提取到图片 URL，chat 摘要: %s", _summarize_payload(current_chat))
            raise HTTPException(status_code=500, detail="Image generation succeeded but no URL found")

        if response_format == "b64_json":
            data = []
            for url in image_urls[:n]:
                data.append({
                    "b64_json": _download_image_as_base64(url),
                    "revised_prompt": prompt,
                })
        else:
            data = [{"url": url, "revised_prompt": prompt} for url in image_urls[:n]]
        return JSONResponse({"created": int(time.time()), "data": data})

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[T2I] 生成失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if acc is not None:
            client.account_pool.release(acc)
            if chat_id:
                asyncio.create_task(client.delete_chat(acc.token, chat_id))
