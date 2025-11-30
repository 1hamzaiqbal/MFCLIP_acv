from pathlib import Path
import requests
import json
import base64

def load_prompt(path):
    return Path(path).read_text().strip()

def load_prompts_for_dataset(name):
    system_path = f"prompts/{name}_system.txt"
    user_path   = f"prompts/{name}_user.txt"
    return load_prompt(system_path), load_prompt(user_path)


def classify_image_qwen(image_path, dataset_name="oxford_pets", openrouter_api_key=""):
    system_prompt, user_prompt = load_prompts_for_dataset(dataset_name)

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json"
    }

    # Encode image
    base64_image = base64.encode_image_to_base64(image_path)
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
