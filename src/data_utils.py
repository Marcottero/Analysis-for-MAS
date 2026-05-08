import requests
import pandas as pd
from bs4 import BeautifulSoup


def normalize_slug_advanced(slug: str, is_aa: bool = False) -> str:
    """Normalize model slug strings.

    This mirrors the original notebook logic, converting various naming
    conventions into a canonical hyphen‑separated form.
    """
    if not isinstance(slug, str) or slug == 'N/D':
        return slug

    s = slug.lower().replace('.', '-')

    if 'claude-haiku-4-5' in s:
        s = s.replace('claude-haiku-4-5', 'claude-4-5-haiku')

    if '-instruct' in s:
        s = s.replace('-instruct', '-it')

    if 'gemma' in s and s.endswith('-it'):
        s = s[:-3]

    for suffix in ['-v1', '-preview']:
        if s.endswith(suffix):
            s = s.replace(suffix, '')

    s = s.replace('-thinking', '-reasoning')

    if is_aa and s.endswith('-reasoning'):
        s = s.replace('-reasoning', '')

    parts = [p for p in s.split('-') if p]
    parts.sort()
    return '-'.join(parts)


def load_openrouter_models(url: str = "https://openrouter.ai/api/v1/models") -> pd.DataFrame:
    """Fetch the OpenRouter model catalogue and return a DataFrame."""
    resp = requests.get(url)
    data = resp.json().get('data', [])
    return pd.json_normalize(data)


def load_whitelist(url: str = "https://cdn.reply.com/documents/challenges/02_26/api_model_whitelist.html") -> list:
    """Download the whitelist HTML and extract model identifiers.

    Returns a list of strings like ``"ai21/jamba-large-1.7"``.
    """
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text(separator='\n')
    whitelist = []
    for line in text.split('\n'):
        line = line.strip()
        if '/' in line and ' ' not in line and not line.startswith('http'):
            whitelist.append(line)
    return whitelist
