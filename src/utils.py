from gensim.parsing.preprocessing import (
    strip_tags,
    strip_multiple_whitespaces,
    strip_punctuation,
    strip_non_alphanum,
)


def clean_text(text):
    text = strip_tags(text)
    text = strip_non_alphanum(text)
    text = strip_multiple_whitespaces(text)
    text = text.lower()

    return text
