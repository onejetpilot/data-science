import re
from typing import Iterable, List

import spacy
from spacy.cli import download as spacy_download
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Базовая модель spaCy для английского
SPACY_MODEL = "en_core_web_sm"


def ensure_spacy_model(model_name: str = SPACY_MODEL):
    """Загрузить модель spaCy; при отсутствии скачать, чтобы запуск был простым."""
    try:
        return spacy.load(model_name, disable=["parser", "ner"])
    except OSError:
        spacy_download(model_name)
        return spacy.load(model_name, disable=["parser", "ner"])


class LemmatizeTransformer(BaseEstimator, TransformerMixin):
    """
    Трансформер, который очищает текст и лемматизирует его через spaCy.
    Использует spaCy pipe для ускорения за счёт батчинга и мультипроцесса.
    """

    def __init__(self, model_name: str = SPACY_MODEL, n_process: int = 2, batch_size: int = 2000):
        self.model_name = model_name
        self.n_process = n_process
        self.batch_size = batch_size
        self._nlp = None

    def fit(self, X, y=None):
        self._nlp = ensure_spacy_model(self.model_name)
        return self

    def transform(self, texts: Iterable[str]) -> List[str]:
        if self._nlp is None:
            self._nlp = ensure_spacy_model(self.model_name)

        cleaned = (re.sub(r"[^a-z\s]", " ", str(text).lower()) for text in texts)
        docs = self._nlp.pipe(
            cleaned,
            n_process=self.n_process,
            batch_size=self.batch_size,
            disable=["parser", "ner"],
        )

        lemmatized = []
        for doc in docs:
            tokens = [
                token.lemma_
                for token in doc
                if not token.is_stop and not token.is_space and len(token) > 2
            ]
            lemmatized.append(" ".join(tokens))

        return lemmatized


def build_pipeline():
    """Создать sklearn-пайплайн: лемматизация, TF-IDF и LinearSVC."""
    return Pipeline(
        steps=[
            ("lemmatizer", LemmatizeTransformer()),
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
            (
                "clf",
                LinearSVC(
                    class_weight="balanced",
                    max_iter=20000,
                    C=0.25,
                    loss="squared_hinge",
                    random_state=42,
                ),
            ),
        ]
    )

