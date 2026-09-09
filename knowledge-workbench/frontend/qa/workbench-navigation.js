async (page) => {
  const assert = (ok, message) => { if (!ok) throw new Error(message) };
  const empty = { nodes: [], edges: [], total_nodes: 0, total_edges: 0, truncated: false, warnings: [] };
  const preview = { ...empty, nodes: [{ id: 'overview', label: '整体预览实体', type: 'person', description: '', retrieved: false }], total_nodes: 1 };
  const record = { id: 'qa-history', created_at: '2026-09-06T00:00:00Z', query: '历史记录测试问题', method_id: 'lightrag', options: { mode: 'mix' }, top_k: 5, status: 'success', answer: '历史记录测试答案',
    latency_ms: 20, retrieval_ms: 10, generation_ms: 10, error: null,
    retrieval: { graph: { ...preview, nodes: [{ id: 'history', label: '历史命中实体', type: 'person', description: '', retrieved: true }] }, chunks: [{ content: '历史测试证据' }], entities: [], relationships: [], references: [], context_text: '历史测试证据', warnings: [], metadata: {} } };
  let graphLoads = 0, submissions = [];
  await page.route('**/api/**', async route => {
    const path = '/api/' + route.request().url().split('/api/')[1];
    let json;
    if (path === '/api/health') json = { api_key_configured: true, llm_model: 'qa', embedding_model: 'qa' };
    else if (path === '/api/dataset/status') json = { dataset: { corpus_name: 'UI-QA', subset: 'medical', character_count: 100, word_count: 20 } };
    else if (path === '/api/index/status') json = { status: 'ready', reusable: true, error_type: null };
    else if (path === '/api/retrieval/methods') json = { methods: [{ id: 'lightrag', name: 'LightRAG', description: '', options: [{key: 'mode', label: '检索模式', choices: ['local','global','hybrid','mix','naive'], default: 'mix'}] }] };
    else if (path === '/api/runs') json = { runs: [record] };
    else if (path === '/api/runs/qa-history') json = record;
    else if (path === '/api/graph') { graphLoads++; json = preview; }
    else if (path === '/api/query') {
      const payload = route.request().postDataJSON(); submissions.push(payload);
      json = { ...record, query: payload.query, options: payload.options, answer: '纯向量测试答案', retrieval: { ...record.retrieval, graph: empty } };
    } else return route.fulfill({ status: 404, json: {} });
    await route.fulfill({ json });
  });
  try {
    await page.reload();
    await page.getByRole('combobox', {name: '定位图中实体'}).selectOption('overview');
    await page.getByRole('button', {name: '查询记录'}).click();
    await page.getByRole('button', {name: /历史记录测试问题/}).click();
    await page.getByText('历史记录测试答案', {exact: true}).waitFor();
    const before = graphLoads;
    await page.getByRole('button', {name: '检索工作台', exact: true}).click();
    await page.getByRole('heading', {name: '已建图谱预览', exact: true}).waitFor({timeout: 3000});
    await page.getByRole('heading', {name: '从一个好问题开始', exact: true}).waitFor();
    assert(await page.getByRole('textbox', {name: '提出你的问题'}).inputValue() === '', 'Returning must clear the historical question');
    await page.getByRole('combobox', {name: '定位图中实体'}).selectOption('overview');
    assert(graphLoads > before, 'Returning must reload the overview graph');
    assert(submissions.length === 0, 'Returning must never submit a paid query');
    await page.getByRole('combobox', {name: '检索模式', exact: true}).selectOption({ label: 'vector（纯向量）' });
    await page.getByRole('textbox', {name: '提出你的问题'}).fill('纯向量测试问题');
    await page.getByRole('button', {name: '开始查询'}).click();
    await page.getByText('纯向量测试答案', {exact: true}).waitFor();
    assert(submissions.length === 1 && submissions[0].options.mode === 'naive', 'Vector label must submit native naive mode');
    await page.getByRole('heading', {name: '本次没有可展示的图节点'}).waitFor();
    await page.getByRole('tab', {name: '文本片段 1'}).waitFor();
    await page.getByRole('button', {name: '检索工作台', exact: true}).click();
    await page.getByRole('heading', {name: '已建图谱预览', exact: true}).waitFor();
    return { passed: ['history return resets result', 'overview refetched', 'no resubmission', 'vector uses naive', 'text-only results', 'query return resets result'] };
  } finally {
    await page.unroute('**/api/**');
    await page.reload();
  }
}
