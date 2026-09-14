import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from app.medical.retrievers import AdaptiveRetriever
from app.retrieval.contracts import RetrievalRequest
from app.retrieval.registry import RetrievalRegistry


@pytest.mark.asyncio
async def test_bge_routes_encoded_features_once_and_preserves_probabilities():
    features = np.ones((1, 1024), dtype=np.float32) / 32
    calls = []
    async def encode(question):
        calls.append(question)
        return features
    class Predictor:
        classes_ = np.array(['pathrag', 'vector', 'lightrag'])
        def predict(self, value):
            assert value is features
            return ['pathrag']
        def predict_proba(self, value):
            assert value is features
            return np.array([[.7, .2, .1]])
    adapter = AdaptiveRetriever(RetrievalRegistry(), lambda: Predictor(), encode,
                                'bge_m3_cls_logistic_regression')
    request, metadata = await adapter.route(RetrievalRequest(query='Which treatment?', method_id='adaptive'))
    assert calls == ['Which treatment?']
    assert request.method_id == 'pathrag'
    assert metadata['router_kind'] == 'bge_m3_cls_logistic_regression'
    assert metadata['routing_probabilities'] == {'pathrag': .7, 'vector': .2, 'lightrag': .1}


@pytest.mark.asyncio
async def test_bge_encoder_uses_cls_512_normalizes_and_keeps_retrieval_limit():
    import threading
    import torch
    from app.medical.engine import MedicalEngine
    seen = {}
    class Tokenizer:
        def __call__(self, texts, **kwargs):
            seen.update(kwargs)
            return {'input_ids': torch.ones((1, 2), dtype=torch.long)}
    class Model:
        def __call__(self, **kwargs):
            hidden = torch.zeros((1, 2, 1024))
            hidden[0, 0, 0] = 3
            hidden[0, 0, 1] = 4
            hidden[0, 1, :] = 100
            return SimpleNamespace(last_hidden_state=hidden)
    engine = MedicalEngine(SimpleNamespace(manifest={'source_commit': 'test', 'router': {
        'kind': 'bge_m3_cls_logistic_regression'}}), None)
    engine.embedding = SimpleNamespace(_lock=threading.Lock(), _load=lambda: None,
        _tokenizer=Tokenizer(), _model=Model(), device='cpu', max_length=2048)
    features = await engine.router_features('question')
    assert seen['max_length'] == 512
    assert features.shape == (1, 1024)
    np.testing.assert_allclose(features[0, :2], [.6, .8])
    assert engine.embedding.max_length == 2048
