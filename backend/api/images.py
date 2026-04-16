"""
图片生成接口 — 兼容 OpenAI /v1/images/generations 规范。

底层通过现有直连 HTTP 聊天能力触发千问“生成图像”模式，
不依赖浏览器运行时。
"""
import re
import time
import json
import asyncio
import logging
from fastapi import APIRouter, Request, HTTPException
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


def _extract_image_urls(text: str) -> list[str]:
    urls: list[str] = []

    for u in re.findall(r'!\[.*?\]\((https?://[^\s\)]+)\)', text):
        urls.append(u.rstrip(").,;"))

    for u in re.findall(r'"(?:url|image|src|imageUrl|image_url)"\s*:\s*"(https?://[^"]+)"', text):
        urls.append(u)

    cdn_pattern = r'https?://(?:cdn\.qwenlm\.ai|wanx\.alicdn\.com|img\.alicdn\.com|[^\s"<>]+\.(?:jpg|jpeg|png|webp|gif))(?:[^\s"<>]*)'
    for u in re.findall(cdn_pattern, text, re.IGNORECASE):
        urls.append(u.rstrip(".,;)\"'>"))

    seen: set[str] = set()
    result: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def _resolve_image_model(requested: str | None) -> str:
    if not requested:
        return DEFAULT_IMAGE_MODEL
    return IMAGE_MODEL_MAP.get(requested, DEFAULT_IMAGE_MODEL)


def _get_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key", "").strip()


def _build_image_prompt(prompt: str) -> str:
    return (
        "请直接生成图片，不要只输出文字描述。"
        "如果可以生成图片，请返回可访问的图片链接或包含图片链接的结果。\n\n"
        f"用户需求：{prompt}"
    )


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

    log.info(f"[T2I] model={model}, n={n}, prompt={prompt[:80]!r}")

    acc = None
    chat_id = None
    try:
        prompt_text = _build_image_prompt(prompt)
        event_payloads: list[str] = []
        async for item in client.chat_stream_events_with_retry(model, prompt_text, has_custom_tools=False):
            if item.get("type") == "meta":
                acc = item.get("acc")
                chat_id = item.get("chat_id")
                continue
            if item.get("type") != "event":
                continue
            event_payloads.append(json.dumps(item.get("event", {}), ensure_ascii=False))

        if acc is None or chat_id is None:
            raise HTTPException(status_code=500, detail="Image generation session was not created")

        chats = await client.list_chats(acc.token, limit=20)
        current_chat = next((c for c in chats if isinstance(c, dict) and c.get("id") == chat_id), None)
        answer_text = "\n".join(event_payloads)
        if current_chat:
            answer_text += "\n" + json.dumps(current_chat, ensure_ascii=False)
        image_urls = _extract_image_urls(answer_text)
        log.info(f"[T2I] 提取到 {len(image_urls)} 张图片 URL: {image_urls}")

        if not image_urls:
            raise HTTPException(status_code=500, detail="Image generation succeeded but no URL found")

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
