import re
import numpy as np

def tokenize_with_spans(text: str):
    """
    Tokenizes input text into non-whitespace words while tracking exact (start, end) character spans.
    """
    tokens = []
    spans = []
    for match in re.finditer(r'\S+', text):
        tokens.append(match.group())
        spans.append((match.start(), match.end()))
    return tokens, spans

def extract_token_features(tokens: list, i: int) -> dict:
    """
    Feature engineering for token classification:
    Extracts lexical, morphological, orthographic, character, and contextual features.
    """
    token = tokens[i]
    clean_token = re.sub(r'^[^\w]+|[^\w]+$', '', token)
    lower = token.lower()
    clean_lower = clean_token.lower()
    
    features = {
        'token.lower': lower,
        'clean_lower': clean_lower,
        'len': len(token),
        'clean_len': len(clean_token),
        'is_title': token.istitle(),
        'clean_is_title': clean_token.istitle() if clean_token else False,
        'is_upper': token.isupper(),
        'is_lower': token.islower(),
        'is_digit': token.isdigit(),
        'has_digit': any(c.isdigit() for c in token),
        'has_at': '@' in token,
        'has_dot': '.' in token,
        'has_hyphen': '-' in token,
        'has_slash': '/' in token,
        'has_plus': '+' in token,
        'has_paren': '(' in token or ')' in token,
        'is_url_like': bool(re.search(r'(https?://|www\.|\.com|\.org|\.net|\.edu|\.gov|\.io|\.co)', lower)),
        'prefix_2': lower[:2],
        'prefix_3': lower[:3],
        'suffix_2': lower[-2:],
        'suffix_3': lower[-3:],
    }
    
    # Previous token features (window -1)
    if i > 0:
        prev = tokens[i - 1]
        prev_clean = re.sub(r'^[^\w]+|[^\w]+$', '', prev)
        features.update({
            'prev.lower': prev.lower(),
            'prev_clean.lower': prev_clean.lower(),
            'prev.is_title': prev.istitle(),
            'prev.is_digit': prev.isdigit(),
            'prev_has_digit': any(c.isdigit() for c in prev),
            'prev_has_at': '@' in prev,
        })
    else:
        features['BOS'] = True

    # Previous 2 token features (window -2)
    if i > 1:
        prev2 = tokens[i - 2]
        prev2_clean = re.sub(r'^[^\w]+|[^\w]+$', '', prev2)
        features['prev2.lower'] = prev2.lower()
        features['prev2_clean.lower'] = prev2_clean.lower()
        features['prev2.is_title'] = prev2.istitle()

    # Next token features (window +1)
    if i < len(tokens) - 1:
        nxt = tokens[i + 1]
        nxt_clean = re.sub(r'^[^\w]+|[^\w]+$', '', nxt)
        features.update({
            'next.lower': nxt.lower(),
            'next_clean.lower': nxt_clean.lower(),
            'next.is_title': nxt.istitle(),
            'next.is_digit': nxt.isdigit(),
            'next_has_digit': any(c.isdigit() for c in nxt),
            'next_has_at': '@' in nxt,
        })
    else:
        features['EOS'] = True

    # Next 2 token features (window +2)
    if i < len(tokens) - 2:
        nxt2 = tokens[i + 2]
        nxt2_clean = re.sub(r'^[^\w]+|[^\w]+$', '', nxt2)
        features['nxt2.lower'] = nxt2.lower()
        features['nxt2_clean.lower'] = nxt2_clean.lower()
        features['nxt2.is_title'] = nxt2.istitle()

    return features

def ensure_int32_csr(matrix):
    """
    Ensures sparse matrix indices use 32-bit integers for LinearSVC compatibility.
    """
    matrix = matrix.tocsr()
    if matrix.indices.dtype != np.int32:
        matrix.indices = matrix.indices.astype(np.int32)
    if matrix.indptr.dtype != np.int32:
        matrix.indptr = matrix.indptr.astype(np.int32)
    return matrix
