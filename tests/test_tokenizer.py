
import tokenizer

tokenizer = tokenizer.Tokenizer()

def test_single():
    hlist = ["hello"]
    r = tokenizer.extract_tokens("hello")
    assert len(r) == 1
    assert hlist == r
    r = tokenizer.extract_tokens("hello-")
    assert len(r) == 1
    assert hlist == r
    r = tokenizer.extract_tokens("=hello")
    assert len(r) == 1
    assert hlist == r
    r = tokenizer.extract_tokens("hello*")
    assert len(r) == 1
    assert hlist == r
    r = tokenizer.extract_tokens("naïve")
    assert len(r) == 1
    assert ["naïve"] == r


def test_multi():
    hlist = ["hello", "there"]
    r = tokenizer.extract_tokens("hello there")
    assert len(r) == 2
    assert hlist == r
    # hyphen
    r = tokenizer.extract_tokens("hello-there")
    assert len(r) == 2
    assert hlist == r
    # en dash
    r = tokenizer.extract_tokens(f"hello{chr(8211)}there")
    assert len(r) == 2
    assert hlist == r
    # em dash
    r = tokenizer.extract_tokens(f"hello{chr(8212)}there")
    assert len(r) == 2
    assert hlist == r
    r = tokenizer.extract_tokens("hello & there")
    assert len(r) == 2
    assert hlist == r
    r = tokenizer.extract_tokens("   hello   there   ")
    assert len(r) == 2
    assert hlist == r
    # we do not used - as a sep because hyphenated names
    r = tokenizer.extract_tokens("*hello-there?")
    assert len(r) == 2
    assert hlist == r

