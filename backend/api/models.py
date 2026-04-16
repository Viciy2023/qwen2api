from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from backend.core.config import MODEL_MAP, resolve_model
from backend.services.auth_quota import resolve_auth_context
from backend.services.qwen_client import QwenClient

router = APIRouter()


def _build_model_list_payload() -> dict:
    seen: set[str] = set()
    data: list[dict] = []
    for model_id in MODEL_MAP:
        if model_id in seen:
            continue
        seen.add(model_id)
        data.append({"id": model_id, "object": "model", "owned_by": "qwen2api"})
    return {"object": "list", "data": data}


@router.get("/v1/models")
async def list_models(request: Request):
    app = request.app
    users_db = app.state.users_db
    client: QwenClient = app.state.qwen_client

    auth = await resolve_auth_context(request, users_db)
    token = auth.token
    try:
        upstream_models = await client.list_models(token)
    except Exception:
        upstream_models = []

    if upstream_models:
        return JSONResponse({
            "object": "list",
            "data": [
                {
                    "id": item.get("id") or item.get("model") or item.get("name") or str(item),
                    "object": "model",
                    "owned_by": item.get("owned_by", "qwen2api"),
                }
                for item in upstream_models
            ],
        })
    return JSONResponse(_build_model_list_payload())


@router.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    resolved = resolve_model(model_id)
    if resolved == model_id and model_id not in MODEL_MAP:
        raise HTTPException(status_code=404, detail={"error": {"message": f"Model '{model_id}' not found", "type": "invalid_request_error"}})
    return JSONResponse({"id": model_id, "object": "model", "owned_by": "qwen2api", "resolved_model": resolved})
