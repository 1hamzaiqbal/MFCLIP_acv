from pathlib import Path
import requests
import json
import base64
import re

REPO_ROOT = Path(__file__).resolve().parent

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def load_prompt(path):
    return Path(path).read_text().strip()

def load_prompts_for_dataset(name):
    system_path = REPO_ROOT / "prompts" / f"{name}_system.txt"
    user_path   = REPO_ROOT / "prompts" / f"{name}_user.txt"

    return load_prompt(system_path), load_prompt(user_path)

def fix_json_list(raw_text):
    """
    Attempt to repair common JSON formatting issues from LLM output.
    Only repairs:
      - missing closing ']' or '}'
      - trailing commas
      - dict wrapper {"predictions": [...]}

    Never invents missing values.
    If cannot fix safely → return None.
    """

    txt = raw_text.strip()

    # If it's a dict like {"predictions": [...]}
    if txt.startswith("{") and "predictions" in txt:
        try:
            data = json.loads(txt)
            if isinstance(data, dict) and "predictions" in data:
                return data["predictions"]
        except:
            pass  # fall through to repair attempts

    # Remove trailing commas: [1,2,3,]
    txt = re.sub(r",\s*]", "]", txt)

    # If it starts like a list but ends mid-way
    if txt.startswith("[") and not txt.endswith("]"):
        txt = txt + "]"

    # If it starts like a dict but missing ending
    if txt.startswith("{") and not txt.endswith("}"):
        txt = txt + "}"

    # Final attempt to parse
    try:
        data = json.loads(txt)
        return data
    except:
        return None

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
