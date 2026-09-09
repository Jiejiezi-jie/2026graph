async (page) => {
  // Offline UI QA only. Intercept ALL query requests; never calls paid APIs.
  // No fake records are written to the real backend or left on the final page.
  await page.route('**/api/query', async route => {
    const request = route.request().postDataJSON()
    const empty = request.query === 'EMPTY'
    const failed = request.query === 'FAILURE'
    const graph = {
      nodes: empty ? [] : [
        { id: 'UI-A', label: 'UI 测试实体 A', type: 'person', description: '仅用于浏览器测试。', retrieved: true },
        { id: 'UI-B', label: 'UI 测试实体 B', type: 'person', description: '仅用于浏览器测试。', retrieved: true },
        { id: 'UI-C', label: 'UI 补充邻居', type: 'location', description: '不是检索证据。', retrieved: false },
      ],
      edges: empty ? [] : [
        { id: 'AB', source: 'UI-A', target: 'UI-B', label: 'test', description: '测试命中关系', retrieved: true, directed: false },
        { id: 'BC', source: 'UI-B', target: 'UI-C', label: 'neighbor', description: '展示补充', retrieved: false, directed: false },
      ],
      total_nodes: 3, total_edges: 2, truncated: false, warnings: [],
    }
    await route.fulfill({ json: {
      id: 'browser-test-only', created_at: new Date().toISOString(), query: request.query,
      method_id: request.method_id, options: request.options, top_k: request.top_k,
      status: failed ? 'failure' : 'success', answer: failed ? null : 'UI 测试数据，非模型回答。',
      latency_ms: 25, retrieval_ms: 10, generation_ms: 15,
      error: failed ? { code: 'LLM_TIMEOUT', message: 'UI 测试：模拟生成超时，证据保留。' } : null,
      retrieval: { graph, entities: [], relationships: [], references: [],
        chunks: empty ? [] : [{ chunk_id: 'ui-test-chunk', content: '<script>unsafe()</script> 应当显示为文本' }],
        context_text: empty ? null : 'UI 测试上下文', warnings: [], metadata: { corpus_name: 'UI-TEST' } },
    } })
  })
  try {
    await page.setViewportSize({ width: 1440, height: 1050 })
    const layout = await page.evaluate(() => {
      const answer = document.querySelector('.answer-panel')
      const graph = document.querySelector('.graph-panel')
      const evidence = document.querySelector('.evidence-panel')
      const style = getComputedStyle(answer)
      return {
        answerRadius: style.borderRadius,
        answerTop: style.borderTopWidth,
        graphDivider: getComputedStyle(graph).borderLeftWidth,
        evidenceRadius: getComputedStyle(evidence).borderRadius,
        graphWidth: graph.getBoundingClientRect().width,
        answerWidth: answer.getBoundingClientRect().width,
      }
    })
    if (layout.answerRadius !== '0px' || layout.answerTop !== '0px' || layout.evidenceRadius !== '0px')
      throw new Error('Workspace sections must not be rounded cards: ' + JSON.stringify(layout))
    if (layout.graphDivider !== '1px' || layout.graphWidth <= layout.answerWidth)
      throw new Error('Graph must be the larger continuous pane with one divider')
    for (const mode of ['local', 'global', 'hybrid', 'mix']) {
      await page.getByRole('combobox', { name: '检索模式', exact: true }).selectOption(mode)
      await page.getByRole('textbox', { name: '提出你的问题' }).fill('UI-TEST ' + mode)
      await page.getByRole('button', { name: '开始查询' }).click()
      await page.getByText('UI 测试数据，非模型回答。', { exact: true }).waitFor()
      await page.getByRole('tab', { name: '文本片段 1' }).waitFor()
      await page.getByRole('table', { name: '文本片段检索结果' }).waitFor()
    }
    await page.getByRole('combobox', { name: '定位图中实体' }).selectOption('UI-A')
    await page.getByRole('heading', { name: 'UI 测试实体 A' }).waitFor()
    await page.getByRole('checkbox', { name: '仅命中' }).check()
    await page.getByRole('table', { name: '文本片段检索结果' }).getByText('查看原始字段').click()
    await page.getByRole('definition').filter({ hasText: '<script>unsafe()</script> 应当显示为文本' }).waitFor()
    await page.screenshot({ path: 'output/playwright/query-mocked.png', fullPage: true })
    await page.getByRole('textbox', { name: '提出你的问题' }).fill('FAILURE')
    await page.getByRole('button', { name: '开始查询' }).click()
    await page.getByText('UI 测试：模拟生成超时，证据保留。').waitFor()
    await page.getByRole('tab', { name: '文本片段 1' }).waitFor()
    await page.getByRole('textbox', { name: '提出你的问题' }).fill('EMPTY')
    await page.getByRole('button', { name: '开始查询' }).click()
    await page.getByRole('tab', { name: '文本片段 0' }).waitFor()
    if (await page.getByRole('heading', { name: 'UI 测试实体 A' }).count()) throw new Error('Stale graph detail')
    await page.getByRole('button', { name: '管理数据与索引' }).click()
    await page.getByRole('dialog').waitFor()
    await page.getByRole('button', { name: '关闭数据管理' }).click()
    await page.setViewportSize({ width: 375, height: 812 })
    if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw new Error('Mobile horizontal overflow')
    await page.screenshot({ path: 'output/playwright/mobile-mocked.png', fullPage: true })
    return { passed: ['four mode controls', 'answer rendering', 'chunks', 'node selection', 'hit filter', 'generation failure preserves evidence', 'empty result clears detail', 'dialog', '375px no overflow'], paidQueries: 0 }
  } finally {
    await page.unroute('**/api/query')
    await page.setViewportSize({ width: 1440, height: 1050 })
    await page.reload()
  }
}
