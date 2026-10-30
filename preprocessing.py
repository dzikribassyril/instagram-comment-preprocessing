"""Conservative cleaning helpers for the Instagram comment dataset."""

import re

import pandas as pd


# Common emoji blocks, regional-indicator flags, keycaps, and emoji joiners.
_EMOJI_PATTERN = re.compile(
    "["
    "\\U0001F1E6-\\U0001F1FF"
    "\\U0001F300-\\U0001FAFF"
    "\\u2600-\\u27BF"
    "\\uFE0F\\u200D\\u20E3"
    "]+"
)
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_MENTION_PATTERN = re.compile(r"(?<!\w)@\w+")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_comments(
    comments: pd.DataFrame, *, text_column: str = "text", min_length: int = 4
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return cleaned comments and excluded rows, retaining the original text.

    Emoji, URLs, and @mentions are removed. Whitespace is normalized. Rows
    whose cleaned text is empty or shorter than ``min_length`` are excluded.
    """
    if text_column not in comments.columns:
        raise KeyError(f"Kolom teks '{text_column}' tidak ditemukan")
    if min_length < 1:
        raise ValueError("min_length harus minimal 1")

    result = comments.copy()
    original = result[text_column].fillna("").astype(str)
    cleaned = original.map(lambda value: _URL_PATTERN.sub(" ", value))
    cleaned = cleaned.map(lambda value: _MENTION_PATTERN.sub(" ", value))
    cleaned = cleaned.map(lambda value: _EMOJI_PATTERN.sub("", value))
    cleaned = cleaned.map(lambda value: _WHITESPACE_PATTERN.sub(" ", value).strip())
    result["text_clean"] = cleaned

    is_empty = cleaned.eq("")
    is_short = cleaned.str.len().lt(min_length)
    result["exclusion_reason"] = ""
    result.loc[is_empty, "exclusion_reason"] = "empty_after_cleaning"
    result.loc[~is_empty & is_short, "exclusion_reason"] = "short_text"

    excluded = result.loc[result["exclusion_reason"].ne("")].copy()
    retained = result.loc[result["exclusion_reason"].eq("")].drop(
        columns="exclusion_reason"
    ).copy()
    return retained.reset_index(drop=True), excluded.reset_index(drop=True)
