"""Turn an extracted sample into token ids: prompt + forced assistant prefix.

The prompt is rendered with the model's own chat template (what LM Studio does).
For tool-call samples the assistant prefix runs up to the start of the bulk
argument (old_string / content / command), so the model then writes the part a
coding agent spends its time on. For text samples the prefix is just the empty
think block.
"""
import json

TOOLS = [
    {"type": "function", "function": {"name": n, "description": d, "parameters": {
        "type": "object", "properties": {p: {"type": "string"} for p in ps}, "required": ps[:1]}}}
    for n, d, ps in [
        ("Bash", "Run a shell command.", ["command", "description"]),
        ("Read", "Read a file from disk.", ["file_path"]),
        ("Edit", "Replace old_string with new_string in a file.", ["file_path", "old_string", "new_string"]),
        ("Write", "Write a file.", ["file_path", "content"]),
        ("Grep", "Search file contents.", ["pattern", "path"]),
    ]
]
SYSTEM = ("You are Claude Code, an interactive CLI coding agent. Use the tools to read, "
          "edit and run code in the user's project. Be concise.")
CUT = {"edit": "old_string", "write": "content", "bash": "command"}


def render(sample, tokenizer, thinking=False):
    msgs = [{"role": "system", "content": SYSTEM}] + sample["messages"]
    if thinking:  # plain prompt with thinking on: the model writes its reasoning first, no forced prefix
        text = tokenizer.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, tokenize=False,
                                             enable_thinking=True)
        return tokenizer.encode(text, add_special_tokens=False), ""
    prompt = tokenizer.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True,
                                           tokenize=False, enable_thinking=False)
    full = tokenizer.apply_chat_template(msgs + [sample["target"]], tools=TOOLS,
                                         tokenize=False, enable_thinking=False)
    # The template renders past assistant turns without thinking, so the target
    # renders as plain text after the generation prompt (which, with thinking
    # off, already contains the empty think block).
    target = full[len(prompt.rstrip()) :] if full.startswith(prompt.rstrip()) else None
    if target is None:
        raise ValueError("template prefix mismatch")
    kind = sample["kind"]
    if kind in CUT:
        marker = f"<parameter={CUT[kind]}>\n"
        j = target.find(marker)
        if j < 0:
            raise ValueError("marker not found")
        prefix = target[: j + len(marker)]
    else:
        prefix = ""
    text = prompt.rstrip() + prefix if prefix else prompt
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids, target[len(prefix):] if prefix else target


if __name__ == "__main__":
    import sys
    from mlx_lm.tokenizer_utils import load as load_tok
    from pathlib import Path
    tok = load_tok(Path(sys.argv[1]))
    for line in open("data/samples.jsonl"):
        s = json.loads(line)
        try:
            ids, rest = render(s, tok)
        except ValueError as e:
            print(s["id"], "ERR", e)
            continue
        print(s["id"], len(ids), repr(tok.decode(ids[-40:]))[:160], "| rest:", repr(rest[:60]))
