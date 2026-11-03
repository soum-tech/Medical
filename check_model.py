
from pathlib import Path
import torch

from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
)

MODEL_PATH = Path("models/Qwen3.5-0.8B").resolve()

assert (MODEL_PATH / "config.json").exists(), (
    f"Model config not found: {MODEL_PATH}"
)

print("Loading processor...")
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

print("Loading Qwen3.5-0.8B on CPU...")
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_PATH,
    device_map="cpu",
    local_files_only=True,
)

model.eval()

print("Model loaded successfully!")
print("Device:", model.device)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Reply with exactly: MODEL_READY"
            }
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to("cpu")

print("Generating response...")

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
    )

response = processor.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True,
)

print("\nModel response:")
print(response)
