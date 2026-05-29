import torch
from typing import Optional


def hand_test_repl(token_model: Optional[torch.nn.Module],
                   line_model:  Optional[torch.nn.Module],
                   tokenizer,
                   hf_tok,
                   device: torch.device,
                   supports_multiline: bool = False):
    is_hf_line = hf_tok is not None
    cmds = []
    if token_model is not None:
        cmds.append(":token <prefix>")
    if line_model is not None:
        cmds.append(":line <prefix>")
    if supports_multiline and line_model is not None:
        cmds.append(":multiline")
    cmds += [":temp <float>", ":k <int>", ":quit"]

    print("  Python Autocomplete — Interactive Test")
    print("  Commands: " + " | ".join(cmds))
    if line_model is not None:
        print(f"  (line model: {'HuggingFace seq2seq' if is_hf_line else 'custom LineModel'})")
    print()

    temperature = 0.3
    top_k = 10

    def gen_line(prefix: str) -> str:
        if is_hf_line:
            inp = hf_tok(prefix, return_tensors="pt").to(device)
            with torch.no_grad():
                out = line_model.generate(
                    **inp,
                    max_new_tokens=64,
                    temperature=temperature,
                    do_sample=(temperature > 0.1),
                    top_k=top_k,
                    num_beams=1,
                    pad_token_id=hf_tok.pad_token_id or hf_tok.eos_token_id,
                )
            return hf_tok.decode(out[0], skip_special_tokens=True)
        else:
            prefix_ids = tokenizer.encode(prefix)
            if prefix_ids and prefix_ids[-1] == tokenizer.eos_id:
                prefix_ids = prefix_ids[:-1]
            with torch.no_grad():
                out_ids = line_model.generate(
                    prefix_ids,
                    max_new=64,
                    temperature=temperature,
                    top_k=top_k,
                    tokenizer=tokenizer,
                )
            return tokenizer.decode(out_ids)

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

        if raw.startswith(":temp"):
            try:
                temperature = float(raw.split()[1])
                print(f"temperature = {temperature}")
            except (IndexError, ValueError):
                print("Usage: :temp 0.7")
            continue

        if raw.startswith(":k"):
            try:
                top_k = int(raw.split()[1])
                print(f"top_k = {top_k}")
            except (IndexError, ValueError):
                print("Usage: :k 40")
            continue

        if raw.startswith(":multiline"):
            if not (supports_multiline and line_model is not None):
                print("This model does not support multiline mode")
                continue
            print(" Multi-line context. Enter lines; blank line to submit.")
            buf = []
            while True:
                try:
                    line = input(".. ")
                except (EOFError, KeyboardInterrupt):
                    break
                if not line:
                    break
                buf.append(line)
            if not buf:
                continue
            prefix = "\n".join(buf)
            completion = gen_line(prefix)
            print(f"  ← line completion:\n\033[33m{prefix}\033[32m{completion}\033[0m\n")
            continue

        want_token = raw.startswith(":token") or (line_model is None and token_model is not None)
        if want_token and token_model is not None:
            prefix = raw[len(":token"):].strip() if raw.startswith(":token") else raw
            ids = tokenizer.encode(prefix)
            if ids and ids[-1] == tokenizer.eos_id:
                ids = ids[:-1]
            with torch.no_grad():
                new_ids = token_model.generate(
                    ids, max_new=20, temperature=temperature,
                    top_k=top_k, stop_at_word_end=True, tokenizer=tokenizer)
            completion = tokenizer.decode(new_ids)
            print(f"  ← token completion: {prefix}\033[32m{completion}\033[0m\n")
            continue

        want_line = raw.startswith(":line") or (token_model is None and line_model is not None)
        if want_line and line_model is not None:
            prefix = raw[len(":line"):].strip() if raw.startswith(":line") else raw
            completion = gen_line(prefix)
            print(f"  ← line completion: {prefix}\033[33m{completion}\033[0m\n")
            continue

        print("Unrecognised input. Use :token, :line, :multiline, :temp, :k, or :quit")
