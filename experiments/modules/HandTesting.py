import torch
from typing import Optional

def hand_test_repl(token_model: Optional[torch.nn.Module], line_model: Optional[torch.nn.Module], tokenizer, hf_tok, device: torch.device, supports_multiline: bool = False):
    greetings_text = "  Python Autocomplete — Interactive Test \n"
    greetings_text += "  Commands:" + " :token <prefix> |" if token_model is not None else "" + " :line <prefix>  |" if line_model is not None else "" + " :multiline      |" if supports_multiline and line_model is not None else ""
    greetings_text += "            :temp <float>   | :k <int>        | :quit           |"

    temperature = 0.3
    top_k       = 10

    while True:
        try:
            raw = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not raw:
            continue

        if raw.startswith(":quit"):
            break
        elif raw.startswith(":temp"):
            try:   temperature = float(raw.split()[1])
            except: print("Usage: :temp 0.7")
            print(f"temperature = {temperature}")
            continue
        elif raw.startswith(":k"):
            try:   top_k = int(raw.split()[1])
            except: print("Usage: :k 40")
            print(f"top_k = {top_k}")
            continue
        elif (raw.startswith(":token") and token_model is not None) or line_model is None:
            prefix = raw[6:] if raw.startswith(":token") else raw
            ids = tokenizer.encode(prefix)[:-1]
            with torch.no_grad():
                new_ids = token_model.generate(
                    ids, max_new=20, temperature=temperature,
                    top_k=top_k, stop_at_word_end=True, tokenizer=tokenizer)
            completion = tokenizer.decode(new_ids)
            print(f"  ← token completion: {prefix}\033[32m{completion}\033[0m\n")
        elif (raw.startswith(":line") and line_model is not None) or token_model is None:
            prefix = raw[5:].strip() if raw.startswith(":line") else raw
            inp = hf_tok(prefix, return_tensors="pt").to(device)
            with torch.no_grad():
                out = line_model.generate(
                    **inp,
                    max_new_tokens=64,
                    temperature=temperature,
                    do_sample=(temperature > 0.1),
                    top_k=top_k,
                )
            completion = hf_tok.decode(out[0], skip_special_tokens=True)
            print(f"  ← line  completion: {prefix}\033[33m{completion}\033[0m\n")
        elif raw.startswith(":multiline") and line_model is not None:
            if not supports_multiline:
                print("This model does not support multiline mode")
                continue
            print(' MultiLine Context Model usage. Enter multi-line prefix, blank line to submit.')
            buf = []
            while True:
                line = input(".. ")
                if not line: break
                buf.append(line)
            prefix = "\n".join(buf)
            inp = hf_tok(prefix, return_tensors="pt").to(device)
            with torch.no_grad():
                out = line_model.generate(
                    **inp,
                    max_new_tokens=64,
                    temperature=temperature,
                    do_sample=(temperature > 0.1),
                    top_k=top_k,
                )
            completion = hf_tok.decode(out[0], skip_special_tokens=True)
            print(f"  ← line  completion: \n\033[33m{prefix}\033[32m{completion}\033[0m\n")