import pytest

from forgelm.tokenizer import Tokenizer

CORPUS = (b"the cat sat on the mat. the cat ate the rat. " * 20) + b"a theory of the thing\n\tthen there was that"


def test_with_no_merges_every_byte_is_its_own_token():
    tok = Tokenizer([])
    data = bytes(range(256))
    assert tok.vocab_size == 256
    assert tok.encode(data) == list(data)
    assert tok.decode_bytes(tok.encode(data)) == data


def test_training_reaches_the_requested_vocabulary_and_is_deterministic():
    first, second = Tokenizer.train(CORPUS, 270), Tokenizer.train(CORPUS, 270)
    assert first.vocab_size == 270
    assert first.merges == second.merges


def test_the_most_frequent_pair_is_merged_first():
    tok = Tokenizer.train(b"xy xy xy xy ab ab", 258)
    assert tok.merges[0] == (ord("x"), ord("y"))
    assert tok.vocab[256] == b"xy"


def test_ties_go_to_the_smallest_pair():
    # Three pairs occur twice: (" ", "a"), ("a", "b") and ("c", "d"). The smallest one wins.
    tok = Tokenizer.train(b"cd ab cd ab", 257)
    assert tok.merges == [(ord(" "), ord("a"))]


def test_training_stops_when_nothing_is_left_to_merge():
    tok = Tokenizer.train(b"ab", 400)
    assert tok.merges == [(ord("a"), ord("b"))]
    assert tok.vocab_size == 257


def test_a_vocabulary_smaller_than_the_byte_values_is_refused():
    with pytest.raises(ValueError):
        Tokenizer.train(CORPUS, 100)


@pytest.mark.parametrize(
    "text",
    ["", "a", "   ", "hello world", "  leading and trailing  ", "tabs\tand\nnewlines\r\n",
     "naïve café — 日本語 🙂", "the cat sat on the mat. " * 3, "\n\n\n"],
)
def test_encoding_round_trips(text):
    tok = Tokenizer.train(CORPUS, 320)
    assert tok.decode(tok.encode(text)) == text


def test_encoding_compresses_text_like_the_training_data():
    tok = Tokenizer.train(CORPUS, 320)
    text = "the cat sat on the mat."
    assert len(tok.encode(text)) < len(text) / 2


def test_merges_are_applied_in_the_order_they_were_learned():
    # 256 = "ab", 257 = "abc", 258 = "bc". "abc" must become [257], never [97, 258].
    tok = Tokenizer([(97, 98), (256, 99), (98, 99)])
    assert tok.encode("abc") == [257]
    assert tok.encode("bc") == [258]
    assert tok.encode("abab") == [256, 256]
    assert tok.decode([257, 258, 256]) == "abcbcab"


def test_merges_never_cross_a_chunk_boundary():
    tok = Tokenizer([(98, 32), (32, 97)])
    assert tok.encode("b a") == [98, 257]
    assert tok.encode("  ") == [32, 32]


def test_strings_and_bytes_encode_the_same():
    tok = Tokenizer.train(CORPUS, 300)
    assert tok.encode("the café") == tok.encode("the café".encode("utf-8"))


def test_invalid_utf8_decodes_with_replacement_instead_of_raising():
    assert Tokenizer([]).decode([0xFF, ord("a")]) == "�a"


def test_save_and_load_round_trip(tmp_path):
    tok = Tokenizer.train(CORPUS, 300)
    tok.save(tmp_path / "tokenizer.json")
    loaded = Tokenizer.load(tmp_path / "tokenizer.json")
    assert loaded.merges == tok.merges
    assert loaded.encode("the cat sat") == tok.encode("the cat sat")
