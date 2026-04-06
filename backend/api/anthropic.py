from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
import json
import logging
import asyncio
from backend.services.qwen_client import QwenClient
from backend.services.token_calc import calculate_usage
from backend.services.prompt_builder import build_prompt_with_tools
from backend.core.config import resolve_model

log = logging.getLogger("qwen2api.anthropic")
router = APIRouter()

@router.post("/messages")
async def anthropic_messages(request: Request):
    """
    Claude API 协议转换层 -> 转入 OpenAI/Qwen 统一处理内核
    """
    app = request.app
    users_db = app.state.users_db
    client: QwenClient = app.state.qwen_client
    
    # 鉴权 (完全复原单文件逻辑)
    auth_header = request.headers.get("x-api-key", "")
    token = auth_header

    from backend.core.config import API_KEYS, settings
    admin_k = settings.ADMIN_KEY

    if API_KEYS:
        if token != admin_k and token not in API_KEYS and not token:
            raise HTTPException(status_code=401, detail="Invalid API Key")

    # 获取下游用户处理配额
    users = await users_db.get()
    user = next((u for u in users if u["id"] == token), None)
    if user and user.get("quota", 0) <= user.get("used_tokens", 0):
        raise HTTPException(status_code=402, detail="Quota Exceeded")
        
    body = await request.json()
    model = resolve_model(body.get("model", "claude-3-5-sonnet"))
    messages = body.get("messages", [])
    tools = body.get("tools", [])
    
    # 构造兼容 OpenAI 的消息格式给 Prompt builder
    system_text = body.get("system", "")
    oai_msgs = []
    if system_text:
        oai_msgs.append({"role": "system", "content": system_text})
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        # 处理 Claude 特有的数组形式
        if isinstance(content, list):
            text_blocks = [blk.get("text", "") for blk in content if blk.get("type") == "text"]
            content = "\n".join(text_blocks)
        oai_msgs.append({"role": role, "content": content})
        
    content = build_prompt_with_tools(oai_msgs, tools)
            
    log.info(f"[Anthropic] model={model}, stream=True, tools={[t.get('name') for t in tools]}, prompt_len={len(content)}")

    try:
        events, chat_id, acc = await client.chat_stream_events_with_retry(model, content)
    except Exception as e:
        log.error(f"Anthropic proxy failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    async def generate():
        full_text = ""
        try:
            # 初始 MessageStart
            start_event = {
                "type": "message_start",
                "message": {"id": "msg_123", "type": "message", "role": "assistant", "model": model, "content": []}
            }
            yield f"event: message_start\ndata: {json.dumps(start_event)}\n\n"
            
            for evt in events:
                if evt.get("type") == "delta":
                    text = evt.get("content", "")
                    full_text += text
                    chunk = {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": text}
                    }
                    yield f"event: content_block_delta\ndata: {json.dumps(chunk)}\n\n"
            
            log.info(f"[Anthropic] Request complete. Generated {len(full_text)} characters.")
                    
            usage = calculate_usage(content, full_text)
            stop_event = {"type": "message_stop", "amazon-bedrock-invocationMetrics": usage}
            yield f"event: message_stop\ndata: {json.dumps(stop_event)}\n\n"
            
            users = await users_db.get()
            for u in users:
                if u["id"] == token:
                    u["used_tokens"] += usage["total_tokens"]
                    break
            await users_db.save(users)
            
        finally:
            client.account_pool.release(acc)
            asyncio.create_task(client.delete_chat(acc.token, chat_id))
            
    return StreamingResponse(generate(), media_type="text/event-stream")
