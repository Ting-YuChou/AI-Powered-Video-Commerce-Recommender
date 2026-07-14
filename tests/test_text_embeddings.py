import numpy as np
import torch

from video_commerce.ml.text_embeddings import MultilingualTextEmbedder


class FakeBatch(dict):
    def to(self, _device):
        return self


class FakeTokenizer:
    def __init__(self):
        self.texts = []

    def __call__(self, texts, **_kwargs):
        self.texts = list(texts)
        return FakeBatch(
            input_ids=torch.tensor([[1, 2, 0], [3, 4, 5]]),
            attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
        )


class FakeModel:
    def eval(self):
        return self

    def __call__(self, **_kwargs):
        return type(
            "Output",
            (),
            {
                "last_hidden_state": torch.tensor(
                    [
                        [[1.0, 0.0], [0.0, 1.0], [100.0, 100.0]],
                        [[1.0, 1.0], [1.0, 1.0], [2.0, 0.0]],
                    ]
                )
            },
        )()


def test_multilingual_text_embedder_prefixes_mean_pools_and_normalizes():
    tokenizer = FakeTokenizer()
    embedder = MultilingualTextEmbedder(
        model_id="intfloat/multilingual-e5-small",
        revision="pinned",
        device=torch.device("cpu"),
        tokenizer=tokenizer,
        model=FakeModel(),
    )

    embeddings = embedder.encode(["新品手機", "new phone"], prefix="passage")

    assert tokenizer.texts == ["passage: 新品手機", "passage: new phone"]
    assert np.asarray(embeddings).shape == (2, 2)
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(embeddings), axis=1),
        np.ones(2),
        atol=1e-6,
    )


def test_multilingual_text_embedder_returns_empty_rows_without_model():
    embedder = MultilingualTextEmbedder(
        model_id="intfloat/multilingual-e5-small",
        revision="pinned",
        device=torch.device("cpu"),
    )

    assert embedder.encode([], prefix="query") == []
