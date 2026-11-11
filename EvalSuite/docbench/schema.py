"""docbench: the fixed extraction schema per document type, the prompt (identical for every config), and the
per-field scoring (exact match after normalization). Pre-registered in PREREG.md: change nothing here after the
first comparison run."""
from __future__ import annotations
import json, re, unicodedata

LAWS = ["England and Wales", "Scotland", "Ireland", "Germany", "Austria", "France", "Belgium", "Bulgaria",
        "Netherlands", "Switzerland", "New York", "Delaware", "California", "Texas"]

# (key, kind, prompt type, description)
FIELDS = {
    "invoice": [
        ("vendor", "text", "string", "name of the company that issued the invoice"),
        ("invoice_number", "id", "string", "the invoice number"),
        ("invoice_date", "date", "date", "date the invoice was issued"),
        ("due_date", "date", "date or null", "payment due date as printed; null if no due date is printed "
                                             "(do not calculate one from payment terms)"),
        ("currency", "currency", "string", "ISO 4217 code of the invoice currency, e.g. EUR"),
        ("subtotal", "amount", "number", "net amount of this invoice before tax"),
        ("tax", "amount", "number", "total tax amount of this invoice (the sum of all tax lines; 0 if the tax is zero)"),
        ("total", "amount", "number", "total of this invoice including tax (excluding any previous balance, "
                                      "payments or deposits)"),
        ("po_number", "id", "string or null", "the customer's purchase order number; null if none"),
    ],
    "receipt": [
        ("merchant", "text", "string", "name of the shop or business, as printed at the top of the receipt"),
        ("date", "date", "date", "transaction date"),
        ("time", "time", "string or null", "transaction time as HH:MM in 24-hour format; null if not printed"),
        ("total", "amount", "number", "final amount paid, including tax and tip"),
        ("payment_method", "enum:cash|card|mobile", "\"cash\", \"card\" or \"mobile\"",
         "how it was paid: cash, card, or mobile (Apple Pay, Google Pay and similar)"),
        ("card_last4", "last4", "string or null", "last 4 digits of the payment card; null if not paid by card "
                                                  "or not printed"),
    ],
    "form": [
        ("full_name", "text", "string", "full name of the policyholder / applicant in natural order (first name(s), "
                                        "then surname), including any middle name"),
        ("date_of_birth", "date", "date", "date of birth of the policyholder / applicant"),
        ("policy_number", "id", "string or null", "the policy number; null if none is given"),
        ("street", "text", "string", "street and house number of the policyholder's current address"),
        ("postal_code", "id", "string", "postal code of that address"),
        ("city", "text", "string", "city of that address"),
        ("country", "country", "string or null", "country of that address, in English; null if not printed"),
        ("selected_options", "set", "list of strings", "labels of the ticked boxes in the cover / options section, "
                                                       "as printed; [] if none is ticked"),
    ],
    "statement": [
        ("opening_balance", "amount", "number", "account balance at the start of the statement period"),
        ("closing_balance", "amount", "number", "account balance at the end of the statement period"),
        ("transaction_count", "int", "integer", "number of transactions posted in the period (not counting "
                                                "balance lines or pending items)"),
        ("largest_debit", "amount_abs", "number", "largest single debit (money out) among the posted transactions "
                                                  "of the period, as a positive number"),
    ],
    "contract": [
        ("parties", "set", "list of strings", "legal names of the two contracting parties, as written"),
        ("effective_date", "date", "date", "date on which the agreement takes effect (not a signature date, "
                                           "unless it takes effect on signature)"),
        ("notice_period", "notice", "object or null",
         "notice required to terminate the agreement without cause (ordinary termination / termination for "
         "convenience), as {\"value\": integer, \"unit\": \"days\" | \"weeks\" | \"months\"} in the unit the "
         "document uses; null if it cannot be terminated without cause"),
        ("governing_law", "enum:" + "|".join(LAWS), "string or null",
         "the law that governs the agreement, one of: " + ", ".join(f'"{x}"' for x in LAWS) + "; null if not stated"),
    ],
}

TYPE_NAME = {"invoice": "an invoice", "receipt": "a till receipt", "form": "a filled-in insurance form",
             "statement": "a bank account statement", "contract": "a contract or agreement letter"}


def prompt(doc_type: str, text: str) -> str:
    lines = [f"Extract the fields below from {TYPE_NAME[doc_type]} and return them as one JSON object.", "",
             "Fields:"]
    for key, _, typ, desc in FIELDS[doc_type]:
        lines.append(f'- "{key}" ({typ}): {desc}')
    lines += ["", "Rules:",
              "- Return only the JSON object, with exactly the keys above.",
              "- Dates as YYYY-MM-DD. Amounts as plain numbers with a decimal point, without currency symbols or "
              "thousands separators (e.g. 1234.56).",
              "- Use null for a field that does not appear in the document.",
              "- Copy names, labels and identifiers as written in the document; do not translate them.",
              "- The text comes from OCR, so the layout may be broken and some words misread.", "",
              "Document:", '"""', text, '"""']
    return "\n".join(lines)


# ---- parsing and scoring -------------------------------------------------------------------------------------------

def strip_fence(s: str) -> str:
    s = s.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.S)
    return m.group(1) if m else s


def parse(output: str):
    """(strict_valid, obj): strict = the output (a markdown fence allowed) is one JSON object; obj = that object, or
    the first balanced {...} in the text that parses (None if none)"""
    s = strip_fence(output)
    try:
        o = json.loads(s)
        if isinstance(o, dict):
            return True, o
    except Exception:
        pass
    for i, ch in enumerate(output):
        if ch != "{":
            continue
        depth, instr, esc = 0, False, False
        for j in range(i, len(output)):
            c = output[j]
            if instr:
                esc = (c == "\\" and not esc)
                if c == '"' and not esc:
                    instr = False
                continue
            if c == '"':
                instr = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        o = json.loads(output[i:j + 1])
                        if isinstance(o, dict):
                            return False, o
                    except Exception:
                        pass
                    break
    return False, None


NULLS = {"", "null", "none", "n/a", "na", "not found", "not present", "not stated", "unknown", "-"}
COUNTRY = {"united kingdom": "united kingdom", "uk": "united kingdom", "u.k.": "united kingdom",
           "great britain": "united kingdom", "gb": "united kingdom",
           "united states": "united states", "usa": "united states", "us": "united states", "u.s.": "united states",
           "u.s.a.": "united states", "united states of america": "united states",
           "germany": "germany", "federal republic of germany": "germany",
           "france": "france", "french republic": "france", "bulgaria": "bulgaria", "republic of bulgaria": "bulgaria",
           "ireland": "ireland", "republic of ireland": "ireland", "austria": "austria", "belgium": "belgium",
           "netherlands": "netherlands", "the netherlands": "netherlands", "switzerland": "switzerland"}


def is_null(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip().casefold() in NULLS)


def norm_text(v) -> str:
    s = unicodedata.normalize("NFKC", str(v)).casefold()
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    return " ".join(s.split())


def norm_id(v) -> str:
    s = unicodedata.normalize("NFKC", str(v)).casefold()
    return "".join(ch for ch in s if ch.isalnum())


def to_number(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = re.sub(r"(?i)\b(eur|usd|gbp|bgn|chf|lv|лв)\b\.?|[€$£\s  ]", "", v)
        if re.fullmatch(r"-?\d+(\.\d+)?", s):
            return float(s)
    return None


def norm_time(v):
    if not isinstance(v, str):
        return None
    m = re.fullmatch(r"\s*(\d{1,2})[:.h](\d{2})(?::\d{2})?\s*([ap]\.?m\.?)?\s*", v.casefold())
    if not m:
        return None
    h, mi, ap = int(m.group(1)), int(m.group(2)), m.group(3)
    if ap:
        h = h % 12 + (12 if ap.startswith("p") else 0)
    return f"{h:02d}:{mi:02d}"


def norm_notice(v):
    if isinstance(v, dict):
        val, unit = v.get("value"), v.get("unit")
    elif isinstance(v, str):
        m = re.fullmatch(r"\s*(\d+)\s*(day|week|month)s?\s*", v.casefold())
        if not m:
            return None
        val, unit = m.group(1), m.group(2)
    else:
        return None
    n = to_number(val)
    if n is None or not isinstance(unit, str):
        return None
    unit = unit.strip().casefold().rstrip("s")
    return (int(n), unit) if n == int(n) and unit in ("day", "week", "month") else None


def norm_city(v) -> str:
    s = norm_text(v)
    return s[3:] if s.startswith("гр ") else s


def correct(kind: str, key: str, pred, gold) -> bool:
    """one field: exact match after the normalization of its kind (pre-registered)"""
    if kind == "set":
        if pred is None or pred == "":
            pred = []
        if isinstance(pred, str):
            pred = [pred]
        if not isinstance(pred, list):
            return False
        return {norm_text(x) for x in pred if not is_null(x)} == {norm_text(x) for x in gold}
    if gold is None:
        return is_null(pred)
    if is_null(pred):
        return False
    if kind == "text":
        return (norm_city(pred) == norm_city(gold)) if key == "city" else norm_text(pred) == norm_text(gold)
    if kind == "id":
        return isinstance(pred, (str, int)) and not isinstance(pred, bool) and norm_id(pred) == norm_id(gold)
    if kind == "date":
        return isinstance(pred, str) and pred.strip() == gold
    if kind in ("amount", "amount_abs"):
        x = to_number(pred)
        if x is None:
            return False
        return abs((abs(x) if kind == "amount_abs" else x) - float(gold)) < 0.005
    if kind == "int":
        x = to_number(pred)
        return x is not None and x == int(gold)
    if kind == "currency":
        return isinstance(pred, str) and pred.strip().upper() == gold
    if kind == "time":
        return norm_time(pred) == gold
    if kind.startswith("enum:"):
        return isinstance(pred, str) and norm_text(pred) == norm_text(gold)
    if kind == "last4":
        if isinstance(pred, int) and not isinstance(pred, bool):
            pred = f"{pred:04d}"
        return isinstance(pred, str) and re.sub(r"\D", "", pred)[-4:] == gold and len(re.sub(r"\D", "", pred)) == 4
    if kind == "country":
        p = norm_text(pred)
        return COUNTRY.get(p, COUNTRY.get(str(pred).strip().casefold(), p)) == COUNTRY.get(gold.casefold(),
                                                                                         gold.casefold())
    if kind == "notice":
        return norm_notice(pred) == (int(gold["value"]), gold["unit"].rstrip("s"))
    raise ValueError(kind)


def score_doc(doc: dict, output: str) -> dict:
    strict, obj = parse(output)
    fields = {}
    for key, kind, _, _ in FIELDS[doc["type"]]:
        gold = doc["gold"][key]
        if obj is None:
            ok = False
        else:
            ok = correct(kind, key, obj.get(key), gold)
        fields[key] = bool(ok)
    return dict(id=doc["id"], type=doc["type"], lang=doc["lang"], json_strict=strict, json_any=obj is not None,
                fields=fields, all_correct=all(fields.values()))
