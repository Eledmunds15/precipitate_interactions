import re

def natural_key(string_):
    """Sort strings naturally by numeric parts."""
    return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', str(string_))]

def master_print(msg):
    """Print only from rank 0."""
    print(f"[MASTER] {msg}", flush=True)