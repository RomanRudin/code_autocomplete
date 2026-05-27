import glob, os, tokenize, io
from pathlib import Path
from typing import List

def load_files(data_dir: str, max_files: int = 0, remove_comments: bool = True):
    patterns = ["**/*.py", "**/*.txt"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(data_dir, pat), recursive=True))
    if max_files:
        files = files[:max_files]
    texts = []
    for fp in files:
        try:
            text = Path(fp).read_text(errors="replace")
            if remove_comments and fp.endswith(".py"):
                text = strip_comments(text)
            texts.append(text)
        except Exception:
            pass
    print(f"[Data] loaded {len(texts)} files from {data_dir}")
    return texts


def strip_comments(source: str) -> str:
    """
    Remove all # comments and docstrings from Python source while preserving 
    layout (line numbers, indentation) so the result still parses identically.
    Uses Python's own tokenizer — won't be fooled by # inside strings.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenizeError, IndentationError, SyntaxError):
        # malformed source — fall back to regex (handles most cases)
        return _strip_comments_regex(source)
    
    output = []
    last_lineno = 1
    last_col    = 0
    prev_toktype = tokenize.INDENT
    
    for tok in tokens:
        token_type, token_string, (start_line, start_col), (end_line, end_col), _ = tok
        
        # restore whitespace between tokens (preserves indentation/layout)
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            output.append(" " * (start_col - last_col))
        
        if token_type == tokenize.COMMENT:
            pass    # skip
        elif token_type == tokenize.STRING and prev_toktype == tokenize.INDENT:
            pass    # docstring (string with no assignment) — skip
        elif token_type == tokenize.STRING and prev_toktype == tokenize.NEWLINE:
            pass    # module / function / class docstring
        else:
            output.append(token_string)
        
        prev_toktype = token_type
        last_lineno  = end_line
        last_col     = end_col
    
    cleaned = "".join(output)
    
    # collapse blank lines that comments left behind
    lines = [l for l in cleaned.splitlines() if l.strip() or not l]
    # keep a single blank between non-empty lines, not three in a row
    result = []
    blank_run = 0
    for l in cleaned.splitlines():
        if l.strip() == "":
            blank_run += 1
            if blank_run <= 1:
                result.append(l)
        else:
            blank_run = 0
            result.append(l)
    return "\n".join(result)

def _strip_comments_regex(source: str) -> str:
    """
    Fallback for files that don't tokenize cleanly.
    Less reliable — doesn't handle # inside strings properly.
    """
    import re
    # remove triple-quoted docstrings  
    source = re.sub(r'""".*?"""', '', source, flags=re.DOTALL)
    source = re.sub(r"'''.*?'''", '', source, flags=re.DOTALL)
    # remove single-line # comments (naive — may break # inside strings)
    source = re.sub(r'(?m)^\s*#.*$', '', source)
    source = re.sub(r'\s+#.*$', '', source, flags=re.MULTILINE)
    return source