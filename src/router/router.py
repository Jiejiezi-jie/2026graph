from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def build_router(feature_kind: str, class_weight: str | None, c_value: float, seed: int) -> Pipeline:
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=30_000,
        sublinear_tf=True,
    )
    if feature_kind == "word_char":
        features = FeatureUnion(
            [
                ("word", word),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=3,
                        max_features=20_000,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
    elif feature_kind == "word":
        features = word
    else:
        raise ValueError(feature_kind)
    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight=class_weight,
                    max_iter=2_000,
                    random_state=seed,
                ),
            ),
        ]
    )

