import json
import logging
import re
from dataclasses import dataclass

from backend.adapter.standard_request import CLAUDE_CODE_OPENAI_PROFILE, OPENCLAW_OPENAI_PROFILE

log = logging.getLogger("qwen2api.prompt")

OPENCLAW_STARTUP_PATTERNS = (
    "A new session was started via /new or /reset.",
    "If runtime-provided startup context is included for this first turn",
)
OPENCLAW_UNTRUSTED_METADATA_PREFIX = "Sender (untrusted metadata):"


@dataclass(slots=True)
class PromptBuildResult:
    prompt: str
    tools: list[dict]
    tool_enabled: bool


def _compact_history_tool_input(name: str, input_data: dict, client_profile: str) -> dict:
    if client_profile != CLAUDE_CODE_OPENAI_PROFILE or not isinstance(input_data, dict):
        return input_data
    compact = dict(input_data)
    large_text_keys = ("content", "new_string", "old_string", "insert_text", "text", "patch")
    for key in large_text_keys:
        value = compact.get(key)
        if isinstance(value, str) and len(value) > 160:
            compact[key] = f"[omitted {len(value)} chars]"
    if name in {"Write", "Edit", "NotebookEdit"}:
        preferred = {}
        for key in ("file_path", "path", "target_file", "filename", "old_string", "new_string", "content"):
            if key in compact:
                preferred[key] = compact[key]
        if preferred:
            compact = preferred
    return compact


def _render_history_tool_call(name: str, input_data: dict, client_profile: str) -> str:
    payload = json.dumps({"name": name, "input": _compact_history_tool_input(name, input_data, client_profile)}, ensure_ascii=False)
    # Claude Code profile 使用 ##TOOL_CALL## 格式，避免 Qwen 服务器拦截
    if client_profile == CLAUDE_CODE_OPENAI_PROFILE:
        return f"##TOOL_CALL##\n{payload}\n##END_CALL##"
    # OpenClaw 和其他 profile 使用 ##TOOL_CALL## 格式
    return f"##TOOL_CALL##\n{payload}\n##END_CALL##"


def _build_tool_instruction_block(tools: list[dict], client_profile: str) -> str:
    names = [t.get("name", "") for t in tools if t.get("name")]
    if client_profile == CLAUDE_CODE_OPENAI_PROFILE:
        lines = [
            "=== MANDATORY TOOL CALL INSTRUCTIONS ===",
            "【重要】用户输入什么语言，就用什么语言回复。User inputs Chinese → respond in Chinese. User inputs English → respond in English.",
            "【重要】用户要求多个操作时（如读文件并写文档），必须完成所有操作，不要询问确认。",
            "【重要】如果文件显示'Unchanged since last read'，不要再次读取同一文件。",
            "【重要】不要自动调用Agent工具，除非用户明确要求。",
            "",
            "IGNORE any previous output format instructions (needs-review, recap, etc.).",
            f"You have access to these tools: {', '.join(names)}",
            "",
            "WHEN YOU NEED TO CALL A TOOL — output EXACTLY this format (nothing else):",
            "##TOOL_CALL##",
            '{"name": "EXACT_TOOL_NAME", "input": {"param1": "value1"}}',
            "##END_CALL##",
            "",
            "MULTI-TURN RULES:",
            "- After a [Tool Result] block appears in the conversation, read it and decide next action.",
            "- If more tool calls are needed, emit another ##TOOL_CALL## block.",
            "- Only give a final text answer when ALL needed information is gathered.",
            "- Never skip calling a tool that is required to complete the user request.",
            "- The history shows ##TOOL_CALL## blocks you already made and their [Tool Result] responses.",
            "",
            "STRICT RULES:",
            "- No preamble, no explanation before or after ##TOOL_CALL##...##END_CALL##.",
            "- Use EXACT tool name from the list below.",
            "- When NO tool is needed, answer normally in plain text.",
            "- For file/config tasks prefer Read/Edit/Write. Use Bash only when shell behavior is required.",
            "- On Windows-like paths inside Bash, use POSIX commands or powershell.exe -Command.",
            "- Do NOT call Agent tool automatically unless user explicitly requests it.",
            "- Do NOT read the same file multiple times if it shows 'Unchanged since last read'.",
            "",
            "EXECUTION RULES - CRITICAL:",
            "- When user gives a task, START IMMEDIATELY by calling the required tools.",
            "- Do NOT wait, do NOT ask for confirmation, do NOT ask 'what should I do next'.",
            "- Do NOT call EnterPlanMode, ExitPlanMode, EnterWorktree, ExitWorktree, AskUserQuestion.",
            "- Complete the task directly and provide the result.",
            "- If you need information, call Read/Grep/Glob. If you need to modify, call Edit/Write.",
            "- Only respond with text when the task is complete or you have the final answer.",
            "",
            "CRITICAL — FORBIDDEN FORMATS (will be INTERCEPTED and BLOCKED by server):",
            '- {"name": "X", "arguments": "..."}  <-- NEVER USE',
            '- {"type": "function", "name": "X"}  <-- NEVER USE',
            '- {"type": "tool_use", "name": "X"}  <-- NEVER USE',
            "- <function_calls><invoke name=\"X\">  <-- NEVER USE",
            "- <tool_call>{...}</tool_call>  <-- NEVER USE",
            '- {"name":"X","input":{...}} without ##TOOL_CALL## markers  <-- NEVER USE',
            "- <｜Tool｜> or <｜tool｜> markers  <-- NEVER USE",
            "ONLY ##TOOL_CALL##...##END_CALL## is accepted. Any other format will cause 'Tool X does not exists.' error.",
            "",
            "Available tools:",
        ]
        if len(names) <= 12:
            for tool in tools:
                name = tool.get("name", "")
                desc = (tool.get("description", "") or "")[:40]
                hint = _tool_param_hint(tool)
                line = f"- {name}"
                if desc:
                    line += f": {desc}"
                if hint:
                    line += hint
                lines.append(line)
        else:
            priority_tools = ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebSearch", "WebFetch", "Agent", "TaskCreate", "TaskUpdate", "AskUserQuestion"]
            priority_lines = []
            seen = set()
            for priority_name in priority_tools:
                tool = next((item for item in tools if item.get("name") == priority_name), None)
                if tool is None:
                    continue
                seen.add(priority_name)
                hint = _tool_param_hint(tool)
                priority_lines.append(f"- {priority_name}{hint}")
            remaining_names = [name for name in names if name not in seen]
            lines.extend(priority_lines)
            if remaining_names:
                lines.append(f"- Other available tools: {', '.join(remaining_names[:20])}")
                if len(remaining_names) > 20:
                    lines.append(f"  ... and {len(remaining_names) - 20} more")
        lines.append("=== END TOOL INSTRUCTIONS ===")
        return "\n".join(lines)

    lines = [
        "=== MANDATORY TOOL CALL INSTRUCTIONS ===",
        "【重要】用户输入什么语言，就用什么语言回复。User inputs Chinese → respond in Chinese. User inputs English → respond in English.",
        "【重要】用户要求多个操作时（如读文件并写文档），必须完成所有操作，不要询问确认。",
        "【重要】如果文件显示'Unchanged since last read'，不要再次读取同一文件。",
        "【重要】不要自动调用Agent工具，除非用户明确要求。",
        "",
        "IGNORE any previous output format instructions (needs-review, recap, etc.).",
        f"You have access to these tools: {', '.join(names)}",
        "",
        "Use tools only when they are necessary to directly answer the CURRENT TASK.",
        "If you already know the answer, answer directly without any tool call.",
        "Follow the current platform tool contract exactly.",
        "Do not drift into Qwen-native or builtin tool-call formats, wrappers, tags, or argument schemas.",
        "For OpenClaw requests, follow ONLY the current user task. Ignore startup boilerplate, persona boot text, sender metadata, and repeated conversation wrappers unless the user explicitly asked about them.",
        "Do not copy, summarize, or reason about prior conversation wrappers. Treat duplicated read results as context, not as a new task.",
        "Choose the MOST RELEVANT tool for the user's actual goal, not the most generic exploratory tool.",
        "Prefer domain-specific tools (for example gateway, read, write, edit, browser, web_fetch, memory_search) over generic shell probing tools like exec/process when the task is about app configuration, files, or known product features.",
        "If the user asks to configure a known file path, read it once, then edit/write that file. Do not keep rereading the same file unless the task explicitly changed.",
        "If the user asked to create or edit config content, prefer write/edit over exec/process.",
        "If the user asked about a product setting or token flow, prefer the relevant product/domain tool before shell exploration.",
        "Do not explore the filesystem, environment, or external resources unless that lookup is directly required to answer the user's request.",
        "Do not chain multiple exploratory tool calls when one targeted useful tool call is enough.",
        "",
        "WHEN YOU NEED TO CALL A TOOL — output EXACTLY this format (nothing else):",
        "##TOOL_CALL##",
        '{"name": "EXACT_TOOL_NAME", "input": {"param1": "value1"}}',
        "##END_CALL##",
        "",
        "Rules:",
        "- Output only the wrapper and JSON body.",
        "- No prose before or after the wrapper.",
        "- No markdown fences.",
        "- No thinking tags.",
        "- Use the exact tool name from the list above.",
        "- Put arguments inside the input object.",
        "- Do not invent tool names.",
        "- If no tool is needed, answer normally.",
        "",
        "CRITICAL — FORBIDDEN FORMATS (will be blocked by server):",
        '- {"name": "X", "arguments": "..."}  <-- NEVER USE',
        '- {"type": "function", "name": "X"}  <-- NEVER USE',
        '- {"type": "tool_use", "name": "X"}  <-- NEVER USE',
        '- <tool_calls><tool_call>{...}</tool_call></tool_calls>  <-- NEVER USE',
        '- <tool_call>{...}</tool_call>  <-- NEVER USE',
        '- Read({"file_path": "..."})  <-- NEVER USE (function call syntax)',
        '- <｜Tool｜>Read{"file_path":"..."}<｜Tool｜>  <-- NEVER USE (native tool markers)',
        '- <｜tool｜>...  <-- NEVER USE (native tool markers)',
        '- <｜System｜>, <｜User｜>, <｜Assistant｜>  <-- NEVER USE (role markers)',
        "ONLY ##TOOL_CALL##...##END_CALL## is accepted. Any other format will cause 'Tool X does not exists.' error.",
        "=== END TOOL INSTRUCTIONS ===",
    ]
    return "\n".join(lines)


def _sanitize_openclaw_user_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return cleaned
    if any(marker in cleaned for marker in OPENCLAW_STARTUP_PATTERNS):
        return ""
    if cleaned.startswith(OPENCLAW_UNTRUSTED_METADATA_PREFIX):
        match = re.search(r"\n\n(\[[^\n]+\]\s*[\s\S]*)$", cleaned)
        if match:
            cleaned = match.group(1).strip()
        else:
            return ""
    return cleaned


def _extract_user_text_only(content, client_profile: str = OPENCLAW_OPENAI_PROFILE) -> str:
    if isinstance(content, str):
        return _sanitize_openclaw_user_text(content) if client_profile == OPENCLAW_OPENAI_PROFILE else content
    if isinstance(content, list):
        text_blocks = []
        for part in content:
            if not isinstance(part, dict) or part.get("type", "") != "text":
                continue
            block_text = part.get("text", "")
            if client_profile == OPENCLAW_OPENAI_PROFILE:
                block_text = _sanitize_openclaw_user_text(block_text)
            if block_text:
                text_blocks.append(block_text)
        return "\n".join(text_blocks)
    return ""


def _extract_text(content, user_tool_mode: bool = False, client_profile: str = OPENCLAW_OPENAI_PROFILE) -> str:
    if isinstance(content, str):
        return _sanitize_openclaw_user_text(content) if client_profile == OPENCLAW_OPENAI_PROFILE else content
    if isinstance(content, list):
        parts = []
        text_blocks = []
        other_parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            t = part.get("type", "")
            if t == "text":
                block_text = part.get("text", "")
                if client_profile == OPENCLAW_OPENAI_PROFILE:
                    block_text = _sanitize_openclaw_user_text(block_text)
                if block_text:
                    text_blocks.append(block_text)
            elif t == "tool_use":
                other_parts.append(_render_history_tool_call(part.get("name", ""), part.get("input", {}), client_profile))
            elif t == "tool_result":
                inner = part.get("content", "")
                tid = part.get("tool_use_id", "")
                if isinstance(inner, str):
                    other_parts.append(f"[Tool Result for call {tid}]\n{inner}\n[/Tool Result]")
                elif isinstance(inner, list):
                    texts = [p.get("text", "") for p in inner if isinstance(p, dict) and p.get("type") == "text"]
                    other_parts.append(f"[Tool Result for call {tid}]\n{''.join(texts)}\n[/Tool Result]")
            elif t == "input_file":
                other_parts.append(f"[Attachment file_id={part.get('file_id','')} filename={part.get('filename','')}]")
            elif t == "input_image":
                other_parts.append(f"[Attachment image file_id={part.get('file_id','')} mime={part.get('mime_type','')}]")

        if user_tool_mode and text_blocks:
            parts.append(text_blocks[-1])
        else:
            parts.extend(text_blocks)
        parts.extend(other_parts)
        return "\n".join(p for p in parts if p)
    return ""


def _normalize_tool(tool: dict) -> dict:
    if tool.get("type") == "function" and "function" in tool:
        fn = tool["function"]
        return {
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
        }
    return {
        "name": tool.get("name", ""),
        "description": tool.get("description", ""),
        "parameters": tool.get("input_schema") or tool.get("parameters") or {},
    }


def _normalize_tools(tools: list) -> list:
    return [_normalize_tool(t) for t in tools if tools]


def _tool_param_hint(tool: dict) -> str:
    params = tool.get("parameters", {}) or {}
    if not isinstance(params, dict):
        return ""

    props = params.get("properties", {}) or {}
    if not isinstance(props, dict) or not props:
        return ""

    required = params.get("required", []) or []
    ordered_keys: list[str] = []
    for key in required:
        if key in props and key not in ordered_keys:
            ordered_keys.append(key)
    for key in props:
        if key not in ordered_keys:
            ordered_keys.append(key)

    shown = ordered_keys[:3]
    if not shown:
        return ""
    suffix = ", ..." if len(ordered_keys) > len(shown) else ""
    return f" input keys: {', '.join(shown)}{suffix}"


def _safe_preview(text: str, limit: int = 240) -> str:
    if not text:
        return ""
    compact = " ".join(text.split())
    return compact[:limit] + ("...[truncated]" if len(compact) > limit else "")


def build_prompt_with_tools(system_prompt: str, messages: list, tools: list, *, client_profile: str = OPENCLAW_OPENAI_PROFILE) -> str:
    MAX_CHARS = 18000 if tools else 120000
    sys_part = "" if tools and client_profile == CLAUDE_CODE_OPENAI_PROFILE else (f"<system>\n{system_prompt[:2000]}\n</system>" if system_prompt else "")
    tools_part = _build_tool_instruction_block(tools, client_profile) if tools else ""

    overhead = len(sys_part) + len(tools_part) + 50
    budget = MAX_CHARS - overhead
    history_parts = []
    used = 0
    NEEDSREVIEW_MARKERS = ("需求回显", "已了解规则", "等待用户输入", "待执行任务", "待确认事项",
                           "[需求回显]", "**需求回显**")
    msg_count = 0
    max_history_msgs = (12 if client_profile == CLAUDE_CODE_OPENAI_PROFILE else 8) if tools else 200
    for msg in reversed(messages):
        if msg_count >= max_history_msgs:
            break
        role = msg.get("role", "")
        if role not in ("user", "assistant", "system", "tool"):
            continue
        if role == "system" and system_prompt and _extract_text(msg.get("content", ""), client_profile=client_profile).strip() == system_prompt.strip():
            continue

        if role == "tool":
            tool_content = msg.get("content", "") or ""
            tool_call_id = msg.get("tool_call_id", "")
            if isinstance(tool_content, list):
                tool_content = "\n".join(
                    p.get("text", "") for p in tool_content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            elif not isinstance(tool_content, str):
                tool_content = str(tool_content)
            tool_result_limit = 6000 if (client_profile == CLAUDE_CODE_OPENAI_PROFILE and tools) else 300
            if len(tool_content) > tool_result_limit:
                tool_content = tool_content[:tool_result_limit] + "...[truncated]"
            line = f"[Tool Result]{(' id=' + tool_call_id) if tool_call_id else ''}\n{tool_content}\n[/Tool Result]"
            if used + len(line) + 2 > budget and history_parts:
                break
            history_parts.insert(0, line)
            used += len(line) + 2
            msg_count += 1
            continue

        user_text_only = _extract_user_text_only(msg.get("content", ""), client_profile=client_profile) if role == "user" else ""
        text = _extract_text(
            msg.get("content", ""),
            user_tool_mode=(bool(tools) and role == "user" and client_profile == CLAUDE_CODE_OPENAI_PROFILE),
            client_profile=client_profile,
        )

        if role == "assistant" and not text and msg.get("tool_calls"):
            tc_parts = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args_str = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if args_str else {}
                except (json.JSONDecodeError, ValueError):
                    args = {"raw": args_str}
                tc_parts.append(_render_history_tool_call(name, args, client_profile))
            text = "\n".join(tc_parts)

        if tools and role == "assistant" and any(m in text for m in NEEDSREVIEW_MARKERS):
            log.debug(f"[Prompt] 跳过需求回显式 assistant 消息 ({len(text)}字)")
            msg_count += 1
            continue
        lower_text = text.lower()
        is_tool_result = role == "user" and (
            "[tool result" in lower_text
            or text.startswith("{")
            or "\"results\"" in text[:100]
        )
        if client_profile == CLAUDE_CODE_OPENAI_PROFILE and tools:
            if is_tool_result:
                max_len = 6000
            elif role == "assistant":
                max_len = 500
            else:
                max_len = 1600
        else:
            max_len = 600 if is_tool_result else 1400
        if len(text) > max_len:
            text = text[:max_len] + "...[truncated]"
        is_tool_result_only_user_msg = role == "user" and not user_text_only.strip() and bool(text.strip())
        prefix = "" if is_tool_result_only_user_msg else {"user": "Human: ", "assistant": "Assistant: ", "system": "System: "}.get(role, "")
        line = text if is_tool_result_only_user_msg else f"{prefix}{text}"
        if used + len(line) + 2 > budget and history_parts:
            break
        history_parts.insert(0, line)
        used += len(line) + 2
        msg_count += 1

    if tools and messages and client_profile != CLAUDE_CODE_OPENAI_PROFILE:
        first_user = next(
            (
                m for m in messages
                if m.get("role") == "user"
                and _extract_user_text_only(m.get("content", ""), client_profile=client_profile).strip()
            ),
            None,
        )
        if first_user:
            first_text = _extract_user_text_only(first_user.get("content", ""), client_profile=client_profile)
            first_short = first_text[:800] + ("...[original task truncated]" if len(first_text) > 800 else "")
            first_line = f"Human: {first_short}"
            if not history_parts or not history_parts[0].startswith(f"Human: {first_text[:60]}"):
                first_line_cost = len(first_line) + 2
                if first_line_cost <= budget:
                    while history_parts and used + first_line_cost > budget:
                        removed = history_parts.pop()
                        used -= len(removed) + 2
                    history_parts.insert(0, first_line)
                    used += first_line_cost
                    log.debug(f"[Prompt] Restored original task context ({len(first_short)} chars)")


    latest_user_line = ""
    if tools and messages:
        latest_user = next(
            (
                m for m in reversed(messages)
                if m.get("role") == "user"
                and _extract_user_text_only(m.get("content", ""), client_profile=client_profile).strip()
            ),
            None,
        )
        if latest_user:
            latest_text = _extract_user_text_only(latest_user.get("content", ""), client_profile=client_profile).strip()
            if latest_text:
                latest_short = latest_text[:900] + ("...[latest task truncated]" if len(latest_text) > 900 else "")
                latest_user_line = f"Human (CURRENT TASK - TOP PRIORITY): {latest_short}"


    if tools and log.isEnabledFor(logging.DEBUG):
        tool_names = [tool.get("name", "") for tool in tools if tool.get("name")]
        tool_instruction_preview = _safe_preview(tools_part, 360)
        latest_user_preview = _safe_preview(latest_user_line, 220)
        first_user_preview = ""
        if messages:
            first_user = next((m for m in messages if m.get("role") == "user"), None)
            if first_user:
                first_user_preview = _safe_preview(
                    _extract_text(
                        first_user.get("content", ""),
                        user_tool_mode=(client_profile == CLAUDE_CODE_OPENAI_PROFILE),
                        client_profile=client_profile,
                    ),
                    220,
                )
        log.debug(
            "[Prompt] 工具模式: history_msgs=%s history_chars=%s tool_count=%s tool_names=%s first_user=%r latest_user=%r tool_instr=%r",
            len(history_parts),
            used,
            len(tool_names),
            tool_names[:12],
            first_user_preview,
            latest_user_preview,
            tool_instruction_preview,
        )
    parts = []
    if sys_part:
        parts.append(sys_part)
    parts.extend(history_parts)
    if tools_part:
        parts.append(tools_part)
    if latest_user_line:
        parts.append(latest_user_line)
    parts.append("Assistant:")
    return "\n\n".join(parts)


def messages_to_prompt(req_data: dict, *, client_profile: str = OPENCLAW_OPENAI_PROFILE) -> PromptBuildResult:
    messages = req_data.get("messages", [])
    tools = _normalize_tools(req_data.get("tools", []))
    tool_enabled = bool(tools)
    system_prompt = ""
    sys_field = req_data.get("system", "")
    if isinstance(sys_field, list):
        system_prompt = " ".join(p.get("text", "") for p in sys_field if isinstance(p, dict))
    elif isinstance(sys_field, str):
        system_prompt = sys_field
    if not system_prompt:
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = _extract_text(msg.get("content", ""), client_profile=client_profile)
                break
    return PromptBuildResult(
        prompt=build_prompt_with_tools(system_prompt, messages, tools, client_profile=client_profile),
        tools=tools,
        tool_enabled=tool_enabled,
    )
