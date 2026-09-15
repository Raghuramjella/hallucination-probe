"""Model loading and prompt construction -- shared by generate and extract.

IMPORTANT: prompt construction lives HERE and nowhere else. If generation and
extraction build prompts differently, the activations no longer correspond to
the answer that was graded, and the whole experiment is silently invalid.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import MODEL_ID, LOAD_IN_4BIT

INSTRUCTION = "Answer with just the answer, no explanation."


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # required for correct batched generation

    kwargs = {"device_map": "auto"}
    if LOAD_IN_4BIT:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    return tok, model


def build_prompt(tok, question: str) -> str:
    msgs = [{"role": "user",
             "content": f"{INSTRUCTION}\nQ: {question}\nA:"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
