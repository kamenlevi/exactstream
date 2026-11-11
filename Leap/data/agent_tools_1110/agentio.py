"""agentio.py: tool calls for serve.py (work in progress; not in the 1.1.0 package).

The model's chat template (engine/models/bare/chat_template.jinja) lists tools in the system turn and has the model answer
a call as

    <tool_call>
    <function=NAME>
    <parameter=ARG>
    VALUE
    </parameter>
    </function>
    </tool_call>

with results coming back as `tool` messages. This module turns the three wire formats serve.py speaks into the template's
messages (Anthropic Messages for Claude Code, OpenAI Responses for Codex, OpenAI chat completions for other apps), and
parses the model's streamed text back into text pieces and tool calls.

For a conversation cache in the engine (later), each rendered request also carries (meta()):
  sys_end     the token index where the system turn (system prompt + tool list) ends: the first token of the conversation;
  side        a background call that must not evict the main conversation's cached state: Claude Code's small-model calls
              (titles, command checks: the launcher names that model exactstream-side), quota probes (max_tokens <= 16),
              and a tool-less call beside a live agent session; lane = main / side / sub (a subagent's own conversation,
              model exactstream-sub);
  conv        a one-way key of the client's session id; prefix_key: of the system turn's tokens.
The same history always renders to the same tokens: one assistant message per model turn however a client groups it (a
Responses client sends a turn's calls as separate items, possibly with outputs between them: the calls one response made
are remembered and rendered together), and Claude Code's billing-header line (not for the model) is left out."""
import hashlib, json, re, uuid

START, END = "<tool_call>", "</tool_call>"
THINK = re.compile(r"</?think>")
BILLING = "x-anthropic-billing-header:"
TURNS = {}                                     # call_id -> the response (model turn) that made it


def text_of(c):
    """Plain text of a content field: a string, or a list of text / input_text / output_text / tool_result blocks."""
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        c = [c]
    out = []
    for b in c:
        if isinstance(b, str):
            out.append(b)
        elif isinstance(b, dict):
            if b.get("type") in ("text", "input_text", "output_text"):
                out.append(b.get("text") or "")
            elif b.get("type") == "tool_result":
                out.append(text_of(b.get("content")))
    return "\n".join(x for x in out if x)


def _tool(name, desc, params):
    return {"type": "function", "function": {"name": name, "description": desc or "",
                                             "parameters": params or {"type": "object", "properties": {}}}}


def _args(a):
    if isinstance(a, dict):
        return a
    if isinstance(a, str) and a.strip():
        try:
            v = json.loads(a)
            return v if isinstance(v, dict) else {"value": v}
        except Exception:
            return {"raw": a}
    return {}


def _system_first(sys_parts, msgs):
    """The template accepts a system turn only first: leading system text joins it (later system text is a user turn,
    made by the callers). Consecutive assistant messages merge: one message per model turn."""
    s = "\n\n".join(p for p in sys_parts if p and p.strip())
    out = []
    for m in msgs:
        m = {k: v for k, v in m.items() if not k.startswith("_")}
        if m["role"] == "assistant" and out and out[-1]["role"] == "assistant":
            p = out[-1]
            p["content"] = "\n\n".join(x for x in (p.get("content"), m.get("content")) if x and x.strip())
            if m.get("tool_calls"):
                p["tool_calls"] = (p.get("tool_calls") or []) + m["tool_calls"]
        else:
            out.append(m)
    return ([{"role": "system", "content": s}] if s else []) + out


def from_anthropic(body):
    """Anthropic Messages -> (template messages, tools). thinking / cache_control / metadata / context_management are ignored."""
    tools = [_tool(t["name"], t.get("description"), t.get("input_schema")) for t in body.get("tools") or []
             if isinstance(t, dict) and t.get("name") and "input_schema" in t]
    if (body.get("tool_choice") or {}).get("type") == "none":
        tools = []
    msgs = []
    for m in body.get("messages") or []:
        role, c = m.get("role"), m.get("content")
        if role == "system":
            msgs.append({"role": "user", "content": text_of(c)})
            continue
        if isinstance(c, str):
            msgs.append({"role": "assistant" if role == "assistant" else "user", "content": c})
            continue
        c = c or []
        if role == "assistant":
            msg = {"role": "assistant", "content": "\n".join(b.get("text") or "" for b in c if b.get("type") == "text")}
            calls = [{"id": b.get("id"), "type": "function", "function": {"name": b.get("name"), "arguments": _args(b.get("input"))}}
                     for b in c if b.get("type") == "tool_use"]
            if calls:
                msg["tool_calls"] = calls
            msgs.append(msg)
        else:
            for b in c:
                if b.get("type") == "tool_result":
                    t = text_of(b.get("content"))
                    msgs.append({"role": "tool", "tool_call_id": b.get("tool_use_id"),
                                 "content": ("Error: " + t) if b.get("is_error") else t})
            t = "\n".join(b.get("text") or "" for b in c if b.get("type") == "text")
            if t.strip():
                msgs.append({"role": "user", "content": t})
    sysv = body.get("system")
    if isinstance(sysv, list):                 # Claude Code's first block is a billing-header line, not for the model
        sysv = [b for b in sysv if not (isinstance(b, dict) and (b.get("text") or "").lstrip().startswith(BILLING))]
    return _system_first([text_of(sysv)], msgs), tools


def from_responses(body):
    """OpenAI Responses -> (template messages, tools). Function tools only (namespace / web_search / local_shell are not
    offered to the model); reasoning items, include, store, prompt_cache_key and client_metadata are ignored."""
    tools = [_tool(t.get("name"), t.get("description"), t.get("parameters")) for t in body.get("tools") or []
             if isinstance(t, dict) and t.get("type") == "function" and t.get("name")]
    if body.get("tool_choice") == "none":
        tools = []
    sys_parts, msgs = [body.get("instructions") or ""], []
    inp = body.get("input")
    if isinstance(inp, str):
        inp = [{"type": "message", "role": "user", "content": inp}]
    for it in inp or []:
        ty = it.get("type", "message")
        if ty == "message":
            role, t = it.get("role", "user"), text_of(it.get("content"))
            if role in ("system", "developer"):
                if msgs:
                    msgs.append({"role": "user", "content": t})
                else:
                    sys_parts.append(t)
            elif role == "assistant":
                msgs.append({"role": "assistant", "content": t})
            else:
                msgs.append({"role": "user", "content": t})
        elif ty == "function_call":
            call = {"id": it.get("call_id"), "type": "function",
                    "function": {"name": it.get("name"), "arguments": _args(it.get("arguments"))}}
            turn, host = TURNS.get(it.get("call_id")), None
            if turn:                           # the same response's message, even with outputs sent between its calls
                host = next((m for m in reversed(msgs) if m.get("_turn") == turn), None)
            if host is None and msgs and msgs[-1]["role"] == "assistant":
                host = msgs[-1]
            if host is None:
                host = {"role": "assistant", "content": ""}
                msgs.append(host)
            host.setdefault("tool_calls", []).append(call)
            if turn:
                host["_turn"] = turn
        elif ty == "function_call_output":
            out = it.get("output")
            msgs.append({"role": "tool", "tool_call_id": it.get("call_id"),
                         "content": out if isinstance(out, str) else (text_of(out) if isinstance(out, list) else json.dumps(out))})
    return _system_first(sys_parts, msgs), tools


def from_chat(body):
    """OpenAI chat completions (with or without tools) -> (template messages, tools)."""
    tools = [_tool(t["function"].get("name"), t["function"].get("description"), t["function"].get("parameters"))
             for t in body.get("tools") or [] if isinstance(t, dict) and t.get("type") == "function" and t.get("function")]
    if body.get("tool_choice") == "none":
        tools = []
    sys_parts, msgs = [], []
    for m in body.get("messages") or []:
        role = m.get("role")
        if role in ("system", "developer"):
            if msgs:
                msgs.append({"role": "user", "content": text_of(m.get("content"))})
            else:
                sys_parts.append(text_of(m.get("content")))
        elif role == "assistant":
            msg = {"role": "assistant", "content": text_of(m.get("content"))}
            if m.get("tool_calls"):
                msg["tool_calls"] = [{"id": c.get("id"), "type": "function", "function": {
                    "name": (c.get("function") or {}).get("name"), "arguments": _args((c.get("function") or {}).get("arguments"))}}
                    for c in m["tool_calls"]]
            msgs.append(msg)
        elif role == "tool":
            msgs.append({"role": "tool", "tool_call_id": m.get("tool_call_id"), "content": text_of(m.get("content"))})
        else:
            msgs.append({"role": "user", "content": text_of(m.get("content"))})
    return _system_first(sys_parts, msgs), tools


def render_ids(tok, msgs, tools):
    text = tok.apply_chat_template(msgs, tools=tools or None, add_generation_prompt=True, enable_thinking=False, tokenize=False)
    return tok.encode(text, add_special_tokens=False)


def render(tok, msgs, tools):
    """(prompt token ids, sys_end): sys_end = the index of the first token after the system turn (0 without one)."""
    ids = render_ids(tok, msgs, tools)
    st = tok.convert_tokens_to_ids("<|im_start|>")
    se = 0
    if len(ids) > 1 and ids[0] == st and tok.decode(ids[1:2]).strip() == "system":
        se = next((i for i in range(1, len(ids)) if ids[i] == st), len(ids))
    return ids, se


def remember(turn, call_ids):
    """serve.py: the calls one response made, so a Responses client's history renders them as one turn"""
    for c in call_ids:
        TURNS[c] = turn
    while len(TURNS) > 4096:
        TURNS.pop(next(iter(TURNS)))


def lane(model, tools, max_tokens, agent_live):
    """('main' | 'side' | 'sub', why). agent_live: a main request with tools came in the last 30 minutes."""
    m = (model or "").lower()
    if m.endswith("-sub") or (m.endswith("-side") and tools):
        return "sub", "subagent model id"
    if m.endswith("-side"):
        return "side", "small-model id"
    try:
        if max_tokens is not None and int(max_tokens) <= 16:
            return "side", f"max_tokens {max_tokens}"
    except (TypeError, ValueError):
        pass
    if not tools and agent_live:
        return "side", "no tools beside an agent session"
    return "main", ""


def conv_id(body, headers):
    """a one-way 12-hex key of the client's session id (Claude Code: metadata.user_id's session_id or the session
    header; Codex: prompt_cache_key or the session-id header); None without one"""
    sid, uid = None, (body.get("metadata") or {}).get("user_id") if isinstance(body.get("metadata"), dict) else None
    if isinstance(uid, str):
        try:
            sid = (json.loads(uid) or {}).get("session_id")
        except Exception:
            sid = uid
    sid = sid or body.get("prompt_cache_key") or headers.get("x-claude-code-session-id") or headers.get("session-id")
    return hashlib.sha1(str(sid).encode()).hexdigest()[:12] if sid else None


def meta(body, headers, tools, ids, se, max_tokens, agent_live):
    """what a conversation cache needs from one rendered request (see the module notes)"""
    ln, why = lane(body.get("model"), tools, max_tokens, agent_live)
    return {"sys_end": se, "side": ln == "side", "lane": ln, "why": why, "conv": conv_id(body, headers), "n": len(ids),
            "prefix_key": hashlib.sha1(json.dumps(ids[:se]).encode()).hexdigest()[:12]}


class ToolParser:
    """Streaming parser: feed() the model's text as it arrives, get ("text", s) and ("tool", name, args) events. Text is
    held back only while it could be the start of <tool_call>; a call is emitted once its </tool_call> arrives."""

    def __init__(self, tools):
        self.buf, self.in_call, self.calls = "", False, 0
        self.schemas = {t["function"]["name"]: t["function"].get("parameters") or {} for t in tools or []}

    def feed(self, s):
        self.buf += THINK.sub("", s)
        out = []
        while True:
            if not self.in_call:
                i = self.buf.find(START)
                if i < 0:
                    keep = next((k for k in range(min(len(START) - 1, len(self.buf)), 0, -1) if START.startswith(self.buf[-k:])), 0)
                    emit, self.buf = self.buf[:len(self.buf) - keep], self.buf[len(self.buf) - keep:]
                    self._text(out, emit)
                    return out
                self._text(out, self.buf[:i])
                self.buf, self.in_call = self.buf[i + len(START):], True
            else:
                j = self.buf.find(END)
                if j < 0:
                    return out
                name, args = self.parse(self.buf[:j])
                if name:
                    out.append(("tool", name, args))
                    self.calls += 1
                self.buf, self.in_call = self.buf[j + len(END):], False

    def finish(self):
        out = []
        if self.in_call:
            name, args = self.parse(self.buf)
            if name:
                out.append(("tool", name, args))
                self.calls += 1
        else:
            self._text(out, self.buf)
        self.buf, self.in_call = "", False
        return out

    def _text(self, out, s):
        if s and not (self.calls and not s.strip()):          # no whitespace-only text after a tool call
            out.append(("text", s))

    def parse(self, body):
        m = re.search(r"<function=([^>\n]+)>", body)
        name = m.group(1).strip() if m else None
        args = {}
        for pm in re.finditer(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", body, re.S):
            k = pm.group(1).strip()
            args[k] = self._typed(name, k, pm.group(2))
        return name, args

    def _typed(self, name, k, v):
        prop = ((self.schemas.get(name) or {}).get("properties") or {}).get(k) or {}
        t = prop.get("type")
        if t == "string" or (isinstance(t, list) and t == ["string"]):
            return v
        try:
            return json.loads(v)
        except Exception:
            return v


def new_id(prefix):
    return prefix + uuid.uuid4().hex[:24]
