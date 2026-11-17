"""Load qwen3-8b-chem-v5 on top of Qwen/Qwen3-8B (8B, not 14B)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = "Qwen/Qwen3-8B"
ADAPTER = "odelpa/qwen3-8b-chem-v5"

# Evaluation setup: 4-bit NF4. For a full-precision load, omit quantization_config
# and use torch_dtype=torch.bfloat16.
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    quantization_config=bnb,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, ADAPTER, subfolder="adapter")
# Optional: merge into the base weights (uses more memory; not needed for inference)
# model = model.merge_and_unload()

messages = [{"role": "user", "content": "What is the oxidation state of Fe in Fe2O3?"}]
inputs = tok.apply_chat_template(
    messages,
    add_generation_prompt=True,
    enable_thinking=False,  # non-thinking, as in the evaluation
    return_tensors="pt",
).to(model.device)

out = model.generate(inputs, max_new_tokens=128, do_sample=False)
print(tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
