import json
from types import SimpleNamespace
import pytest
from app.config import Settings
from app.domain.errors import AppError
from app.medical.engine import SessionLLM

@pytest.mark.asyncio
@pytest.mark.parametrize('pathrag',[False,True])
@pytest.mark.parametrize('finish,content,expected',[
 ('stop','{"high_level_keywords":["diagnosis"],"low_level_keywords":["CML"]}',None),
 ('length','', 'TRUNCATED'),('stop','', 'EMPTY_LLM_RESPONSE')])
async def test_keyword_budget_format_diagnostics(monkeypatch,caplog,pathrag,finish,content,expected):
    calls=[]
    class Client:
        def __init__(self,**kwargs): self.chat=SimpleNamespace(completions=self)
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        async def create(self,**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish,message=SimpleNamespace(content=content,reasoning_content='private thinking'))],usage=SimpleNamespace(prompt_tokens=500,completion_tokens=768,completion_tokens_details=SimpleNamespace(reasoning_tokens=768)))
    monkeypatch.setattr('openai.AsyncOpenAI',Client)
    llm=SessionLLM(Settings(deepseek_api_key='test-secret',_env_file=None),pathrag_keywords=pathrag)
    kwargs={'keyword_extraction':True} if pathrag else {'response_format':{'type':'json_object'}}
    with caplog.at_level('INFO',logger='app.medical.engine'):
        if expected:
            with pytest.raises(AppError) as error: await llm.complete('question',**kwargs)
            assert expected in error.value.code
        else: assert json.loads(await llm.complete('question',**kwargs))['low_level_keywords']==['CML']
    assert calls[0]['max_tokens']==4096
    assert calls[0]['response_format']=={'type':'json_object'}
    assert 'finish_reason=' in caplog.text
    assert 'private thinking' not in caplog.text and 'test-secret' not in caplog.text
