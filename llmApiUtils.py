from pathlib import Path
import requests
import json
import base64
import re
import ast

REPO_ROOT = Path(__file__).resolve().parent

def accuracy_calc_for_llm(preds, labels):
    correct = sum(1 for p, gt in zip(preds, labels) if p == gt)
    return correct / len(labels)

def oxford_pets_to_binary(idx):
    """Convert 1–37 class index → 0 (cat) or 1 (dog)."""
    idx = int(idx)
    return 0 if 1 <= idx <= 12 else 1


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def load_prompt(path):
    return Path(path).read_text().strip()

def load_prompts_for_dataset(name):
    system_path = REPO_ROOT / "prompts" / f"{name}_system.txt"
    user_path   = REPO_ROOT / "prompts" / f"{name}_user.txt"

    return load_prompt(system_path), load_prompt(user_path)

def parse_list(raw_text, B):
    """Robust parsing of Python-style list responses."""

    # 1. Try literal_eval directly
    try:
        parsed = ast.literal_eval(raw_text)
        if isinstance(parsed, list):
            return parsed
    except:
        pass

    # 2. Try to auto-fix truncated lists
    #    e.g., "[1,2,3"  → add closing bracket
    fixed = raw_text.strip()
    if not fixed.endswith("]"):
        fixed = fixed + "]"

    try:
        parsed = ast.literal_eval(fixed)
        if isinstance(parsed, list):
            return parsed
    except:
        pass

    # 3. Try removing trailing garbage, only keep inside last []
    if "[" in raw_text:
        sub = raw_text[raw_text.rfind("["):]
        if not sub.endswith("]"):
            sub += "]"

        try:
            parsed = ast.literal_eval(sub)
            if isinstance(parsed, list):
                return parsed
        except:
            pass

    # 4. If STILL broken → fallback
    print("LLM returned invalid list:", raw_text)
    return [0] * B

def classify_image_qwen(image_path, dataset_name="oxford_pets", openrouter_api_key=""):
    system_prompt, user_prompt = load_prompts_for_dataset(dataset_name)

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json"
    }

    # Encode image
    base64_image = encode_image_to_base64(image_path)
    data_url = f"data:image/jpeg;base64,{base64_image}"

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                { "type": "text", "text": user_prompt },
                { "type": "image_url", "image_url": {"url": data_url} }
            ]
        }
    ]

    payload = {
        "model": "qwen/qwen3-vl-30b-a3b-instruct",
        "messages": messages,
        "temperature": 0,
        "max_tokens": 5,
    }

    resp = requests.post(url, headers=headers, json=payload)
    data = resp.json()

    out = data["choices"][0]["message"]["content"].strip()
    return int(out) if out.isdigit() else None
