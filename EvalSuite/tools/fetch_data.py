"""Fetch the public evaluation data into data/raw/ (small files; run with the EvalSuite venv: .venv/bin/python).
MMLU-Pro (TIGER-Lab), GPQA-Diamond (OpenAI simple-evals CSV; the GPQA authors ask that examples not be posted
online: data/ is git-ignored), IFEval (google/IFEval), MATH-500 (HuggingFaceH4), GSM8K (local copy from MoEPrefetch),
long documents (Project Gutenberg, Bulgarian Wikipedia) and the HumanEval+ cache (evalplus)."""
import json, os, shutil, subprocess, sys, urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
URLS = {
    "mmlu_pro_test.parquet": "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet",
    "gpqa_diamond.csv": "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv",
    "ifeval_input_data.jsonl": "https://huggingface.co/datasets/google/IFEval/resolve/main/ifeval_input_data.jsonl",
    "math500_test.jsonl": "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl",
    "doc_en_pride_prejudice.txt": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "doc_fr_monte_cristo.txt": "https://www.gutenberg.org/cache/epub/17989/pg17989.txt",
    "doc_de_der_prozess.txt": "https://www.gutenberg.org/cache/epub/22367/pg22367.txt",
}
BG_TITLES = ["България", "История на България", "София", "Пловдив", "Иван Вазов", "Христо Ботев", "Васил Левски",
             "Рила", "Черно море", "Българска литература", "Първа българска държава", "Втора българска държава",
             "Освобождение на България", "Кирил и Методий", "Българска азбука", "Дунав", "Стара планина", "Родопи"]


def get(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    r = subprocess.run(["curl", "-sfL", "--retry", "3", "-o", path + ".part", url])
    if r.returncode:
        sys.exit(f"download failed: {url}")
    os.replace(path + ".part", path)
    print("got", os.path.basename(path), os.path.getsize(path), flush=True)


def main():
    os.makedirs(RAW, exist_ok=True)
    for name, url in URLS.items():
        get(url, os.path.join(RAW, name))
    shutil.copy(os.path.expanduser("~/MoEPrefetch/data/gsm8k/test.jsonl"), os.path.join(RAW, "gsm8k_test.jsonl"))
    bg = os.path.join(RAW, "doc_bg_wikipedia.txt")
    if not os.path.exists(bg):
        parts = []
        for t in BG_TITLES:
            q = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1, "titles": t,
                                        "format": "json", "redirects": 1})
            r = subprocess.run(["curl", "-sfL", f"https://bg.wikipedia.org/w/api.php?{q}"], capture_output=True)
            pages = json.loads(r.stdout or b"{}").get("query", {}).get("pages", {})
            for p in pages.values():
                if p.get("extract"):
                    parts.append(f"{p['title']}\n\n{p['extract']}")
        open(bg, "w").write("\n\n".join(parts))
        print("got", os.path.basename(bg), os.path.getsize(bg), flush=True)
    from evalplus.data import get_human_eval_plus
    print("humaneval+ problems:", len(get_human_eval_plus()), flush=True)


if __name__ == "__main__":
    main()
