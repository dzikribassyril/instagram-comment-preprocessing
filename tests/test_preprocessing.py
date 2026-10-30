import pandas as pd

from preprocessing import clean_comments


def test_removes_emoji_and_short_comments_but_preserves_original_text():
    source = pd.DataFrame(
        {
            "comment_id": [1, 2, 3, 4],
            "text": ["Bagus sekali 😂", "😂😂", "Ya!", "Tidak puas 😡"],
        }
    )

    cleaned, excluded = clean_comments(source)

    assert cleaned["text_clean"].tolist() == ["Bagus sekali", "Tidak puas"]
    assert cleaned["text"].tolist() == ["Bagus sekali 😂", "Tidak puas 😡"]
    assert excluded["comment_id"].tolist() == [2, 3]
    assert excluded["exclusion_reason"].tolist() == ["empty_after_cleaning", "short_text"]


def test_removes_urls_mentions_and_normalizes_whitespace():
    source = pd.DataFrame({"text": ["  Setuju   @akun lihat https://contoh.id/post  "]})

    cleaned, excluded = clean_comments(source)

    assert excluded.empty
    assert cleaned.loc[0, "text_clean"] == "Setuju lihat"


def test_does_not_drop_negation_or_non_emoji_punctuation():
    source = pd.DataFrame({"text": ["Tidak bagus!!!"]})

    cleaned, excluded = clean_comments(source)

    assert excluded.empty
    assert cleaned.loc[0, "text_clean"] == "Tidak bagus!!!"
