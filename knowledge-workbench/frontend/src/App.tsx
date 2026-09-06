import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react'
import { ArrowUpRight, BookOpen, ChevronDown, ChevronRight, Clock3, Database, FolderOpen, History, Layers2, LoaderCircle, Minus, Plus, Network, Search, Settings2, X } from 'lucide-react'
import Markdown from 'react-markdown'
import GraphPanel from './GraphPanel'
import ApiSettings from './ApiSettings'
import { api, emptyGraph, type Dataset, type GraphData, type Health, type IndexStatus, type Method, type RecordData, type Run } from './api'

const indexLabels: Record<string, string> = { ready: '索引已就绪', not_built: '尚未建立索引', building: '正在建立索引', failed: '索引未完成', interrupted: '上次构建已中断' }
const seconds = (ms: number) => (ms / 1000).toFixed(1) + ' s'
const modeLabel = (methodId: string, mode: string) => methodId === 'lightrag' && mode === 'naive' ? 'vector（纯向量）' : mode
const modeDescriptions: Record<string, string> = {
  local: '实体检索：从问题涉及的人物、地点等实体出发，查找相邻关系与关联原文，适合具体对象的细节问题。',
  global: '关系检索：从主题与高层关键词出发，查找相关关系及实体，适合梳理主要主题和整体联系。',
  hybrid: '双路图谱检索：结合 local 的实体检索与 global 的关系检索，兼顾具体细节和整体联系。',
  mix: '图谱与文本融合：在 hybrid 图谱检索之外，加入原文片段的向量检索，综合两类证据。',
  naive: '纯向量检索：按语义相似度查找原文片段，不使用图谱关系，适合直接从原文寻找答案。',
}
const normalizeTopK = (value: number) => Number.isFinite(value) ? Math.max(1, Math.min(50, Math.trunc(value))) : 5
const textValue = (value: unknown) => typeof value === 'string' ? value : JSON.stringify(value)
const tabs = [['chunks', '文本片段'], ['entities', '实体'], ['relationships', '关系'], ['references', '引用'], ['context', '上下文']] as const

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [dataset, setDataset] = useState<Dataset | null>(null)
  const [index, setIndex] = useState<IndexStatus>({ status: 'not_built', reusable: false, error_type: null })
  const [methods, setMethods] = useState<Method[]>([])
  const [methodId, setMethodId] = useState('')
  const [options, setOptions] = useState<Record<string, string>>({})
  const [topK, setTopK] = useState(5)
  const [query, setQuery] = useState('')
  const [preview, setPreview] = useState<GraphData>(emptyGraph)
  const [result, setResult] = useState<Run | null>(null)
  const [graphExpanded, setGraphExpanded] = useState(false)
  const [history, setHistory] = useState<Run[]>([])
  const [page, setPage] = useState<'workbench' | 'history'>('workbench')
  const [tab, setTab] = useState<(typeof tabs)[number][0]>('chunks')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [setupOpen, setSetupOpen] = useState(false)
  const [apiSettingsOpen, setApiSettingsOpen] = useState(false)
  const [subset, setSubset] = useState('novel')
  const [settingUp, setSettingUp] = useState(false)
  const dialog = useRef<HTMLDialogElement>(null)
  const refreshing = useRef(false)
  const graphRefreshQueued = useRef(false)
  const navigationVersion = useRef(0)
  const method = methods.find(m => m.id === methodId)

  const refresh = useCallback(async (loadGraph = false): Promise<void> => {
    if (refreshing.current) {
      if (loadGraph) graphRefreshQueued.current = true
      return
    }
    refreshing.current = true
    try {
      const [h, d, i, r] = await Promise.all([
        api<Health>('/health'), api<{ dataset: Dataset | null }>('/dataset/status'),
        api<IndexStatus>('/index/status'), api<{ runs: Run[] }>('/runs'),
      ])
      setHealth(h); setDataset(d.dataset); setIndex(i); setHistory(r.runs)
      if (loadGraph && i.reusable) {
        try { setPreview(await api<GraphData>('/graph')) } catch (e) { setError((e as Error).message) }
      } else if (!i.reusable) setPreview(emptyGraph)
    } catch (e) { setError((e as Error).message) }
    finally {
      refreshing.current = false
      if (graphRefreshQueued.current) {
        graphRefreshQueued.current = false
        void refresh(true)
      }
    }
  }, [])

  useEffect(() => {
    void refresh(true)
    api<{ methods: Method[] }>('/retrieval/methods').then(data => {
      setMethods(data.methods)
      if (data.methods[0]) {
        setMethodId(data.methods[0].id)
        setOptions(Object.fromEntries(data.methods[0].options.map(o => [o.key, o.default])))
      }
    }).catch(e => setError(e.message))
  }, [refresh])
  useEffect(() => {
    if (index.status !== 'building') return
    const timer = setInterval(() => void refresh(true), 4000)
    return () => clearInterval(timer)
  }, [index.status, refresh])
  useEffect(() => {
    if (!busy) return
    const start = Date.now()
    const timer = setInterval(() => setElapsed(Math.floor((Date.now() - start) / 1000)), 1000)
    return () => clearInterval(timer)
  }, [busy])
  useEffect(() => {
    if (setupOpen) dialog.current?.showModal()
    else dialog.current?.close()
  }, [setupOpen])

  async function submit(event: FormEvent) {
    event.preventDefault()
    if (!query.trim() || busy || !Number.isInteger(topK) || topK < 1 || topK > 50) return
    setBusy(true); setElapsed(0); setError(''); setResult(null)
    try {
      const data = await api<Run>('/query', { query, method_id: methodId, options, top_k: topK })
      setResult(data); setTab('chunks')
      await refresh()
    } catch (e) { setError((e as Error).message) }
    finally { setBusy(false) }
  }

  async function selectDataset() {
    setSettingUp(true); setError('')
    try {
      await api('/dataset/select', { subset })
      setResult(null); await refresh(true)
    } catch (e) { setError((e as Error).message) }
    finally { setSettingUp(false) }
  }

  async function build(rebuild: boolean) {
    const message = rebuild ? '将把现有 workspace 移到本地备份目录，然后重新建图。此操作会调用 DeepSeek 并产生 API 费用。确认继续？' : '建索引会调用 DeepSeek 抽取实体和关系，并产生 API 费用。确认开始？'
    if (!window.confirm(message)) return
    setSettingUp(true); setError('')
    try {
      await api('/index/build', { rebuild })
      setResult(null); await refresh(); setSetupOpen(false)
    } catch (e) { setError((e as Error).message) }
    finally { setSettingUp(false) }
  }

  function enterWorkbench() {
    if (busy) return
    navigationVersion.current++
    setResult(null); setQuery(''); setTab('chunks'); setError(''); setElapsed(0)
    setGraphExpanded(false)
    setPage('workbench')
    void refresh(true)
  }

  async function openRun(id: string) {
    const version = ++navigationVersion.current
    setError('')
    try {
      const run = await api<Run>('/runs/' + id)
      if (version !== navigationVersion.current) return
      setResult(run); setQuery(run.query); setPage('workbench')
      if (methods.some(m => m.id === run.method_id)) {
        setMethodId(run.method_id); setOptions(run.options); setTopK(run.top_k)
      }
    } catch (e) { if (version === navigationVersion.current) setError((e as Error).message) }
  }

  const graph = result?.retrieval.graph || preview
  const unavailable = busy || !method || (methodId === 'lightrag' && !index.reusable) || index.status === 'building' || !Number.isInteger(topK) || topK < 1 || topK > 50
  const evidence = result?.retrieval

  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#" onClick={e => { e.preventDefault(); enterWorkbench() }}><span className="brand-mark"><Network size={22} /></span><span>知图<span className="brand-en">KNOWLEDGE STUDIO</span></span></a>
      <div className="sidebar-label">WORKSPACE</div>
      <nav aria-label="主导航">
        <button className={page === 'workbench' ? 'nav-link active' : 'nav-link'} disabled={busy} onClick={enterWorkbench}><Search size={18} />检索工作台<ChevronRight size={15} /></button>
        <button className={page === 'history' ? 'nav-link active' : 'nav-link'} onClick={() => { navigationVersion.current++; setPage('history'); void refresh() }}><History size={18} />查询记录<span className="nav-count">{history.length}</span></button>
      </nav>
      <div className="dataset-summary">
        <div className="sidebar-label">CURRENT DATASET</div>
        <h3>{dataset?.corpus_name || '尚未选择文本'}</h3>
        <p>{dataset ? 'GraphRAG-Bench / ' + dataset.subset : '导入一份小文本，开始探索'}</p>
        {dataset && <dl><div><dt>文本字符</dt><dd>{dataset.character_count.toLocaleString()}</dd></div><div><dt>词数</dt><dd>{dataset.word_count.toLocaleString()}</dd></div></dl>}
        <div className={'index-status ' + (index.reusable ? 'ready' : '')}><span className="status-dot" aria-hidden="true" />{index.status === 'ready' && !index.reusable ? '索引不可复用' : indexLabels[index.status] || index.status}</div>
        <button className="dataset-button" disabled={busy} onClick={() => setSetupOpen(true)}><FolderOpen size={15} />管理数据与索引<ArrowUpRight size={14} /></button>
      </div>
      <button className="settings-entry" disabled={busy || settingUp || index.status === 'building'} onClick={() => setApiSettingsOpen(true)}><Settings2 size={16} />API 设置<ArrowUpRight size={14} /></button>
      <div className="sidebar-bottom"><span className="small-dot" />本地工作空间<p>单数据集 · 可插拔检索<br />FastAPI / React / Cytoscape</p><span className="version">COURSE DEMO <span>V1.0</span></span></div>
    </aside>

    <main>
      <header className="topbar"><div><span>知识图谱</span><ChevronRight size={13} /><strong>{page === 'history' ? '查询记录' : '检索工作台'}</strong></div><span className="connection"><i className={health ? 'online' : ''} />{health ? '后端已连接' : '等待后端连接'}</span></header>
      <div className={'main-content' + (graphExpanded && page === 'workbench' ? ' graph-workspace-expanded' : '')}>
        <div className="page-heading"><div><span className="eyebrow">EXPLORE · RETRIEVE · UNDERSTAND</span><h1>{page === 'history' ? '每次探索，都有迹可循。' : '让答案，连接到证据。'}</h1><p>{page === 'history' ? '回看本地保存的查询、耗时与检索结果。' : '选择检索方式，在文本与图谱之间探索知识。'}</p></div><span className="outline-badge"><Database size={14} />{dataset?.corpus_name || 'NO DATASET'}</span></div>
        {error && <div className="alert error" role="alert">{error}<button aria-label="关闭提示" onClick={() => setError('')}><X size={16} /></button></div>}
        {health && !health.api_key_configured && <div className="alert">尚未配置 API Key。可以查看已有图谱；运行查询前请在侧栏「API 设置」中填写并保存。</div>}

        {page === 'history' ? <section className="history-panel"><div className="section-heading"><h2>最近查询</h2><span className="muted">最近 20 条 · 本地保留最多 100 条</span></div>
          {!history.length && <div className="empty-copy"><History size={28} /><h3>还没有查询记录</h3><p>在检索工作台提出第一个问题。</p></div>}
          {history.map(run => <button disabled={busy} className="history-row" key={run.id} onClick={() => void openRun(run.id)}><div><strong>{run.query}</strong><span>{new Date(run.created_at).toLocaleString()} · {run.method_id} / {modeLabel(run.method_id, run.options.mode || 'default')}</span></div><span className={run.status === 'success' ? 'success-text' : 'error-text'}>{run.status === 'success' ? '完成' : '失败'}</span><span>{seconds(run.latency_ms)}</span><ArrowUpRight size={16} /></button>)}
        </section> : <>
          <form className="query-composer" onSubmit={submit}>
            <label className="query-label" htmlFor="query">提出你的问题</label>
            <textarea id="query" maxLength={4000} value={query} disabled={busy} onChange={e => setQuery(e.target.value)} placeholder="例如：故事中的核心人物是谁？他们之间有哪些重要关系？" onKeyDown={e => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); if (!unavailable) void submit(e) } }} />
            <div className="query-controls">
              <fieldset className="retrieval-parameters"><legend className="sr-only">检索参数</legend>
              <div className="control method-control"><label htmlFor="method">检索方法</label><Network className="parameter-icon" size={18} strokeWidth={1.5} aria-hidden="true" /><div className="parameter-select"><select id="method" value={methodId} disabled={busy} onChange={e => { setMethodId(e.target.value); const m = methods.find(item => item.id === e.target.value); setOptions(Object.fromEntries((m?.options || []).map(o => [o.key, o.default]))) }}>{!methods.length && <option value="">加载方法…</option>}{methods.map(m => <option value={m.id} key={m.id}>{m.name}</option>)}</select><ChevronDown size={14} strokeWidth={1.5} aria-hidden="true" /></div></div>
              {method?.options.map(o => <div className="control" key={o.key}><label htmlFor={'opt-' + o.key}>{o.label}</label><Layers2 className="parameter-icon" size={18} strokeWidth={1.5} aria-hidden="true" /><div className="parameter-select"><select id={'opt-' + o.key} disabled={busy} value={options[o.key] || o.default} onChange={e => setOptions({ ...options, [o.key]: e.target.value })}>{o.choices.map(c => <option key={c} value={c}>{o.key === 'mode' ? modeLabel(methodId, c) : c}</option>)}</select><ChevronDown size={14} strokeWidth={1.5} aria-hidden="true" /></div></div>)}
              <div className="control topk"><label htmlFor="topk">Top-K</label><div className="topk-stepper">
                <button type="button" aria-label="减少 Top-K" disabled={busy || topK <= 1} onClick={() => setTopK(value => normalizeTopK(normalizeTopK(value) - 1))}><Minus size={14} strokeWidth={1.5} /></button>
                <input id="topk" type="number" inputMode="numeric" min={1} max={50} step={1} required disabled={busy} value={Number.isFinite(topK) ? topK : ''} onChange={e => setTopK(e.target.value === '' ? NaN : Number(e.target.value))} onBlur={() => setTopK(normalizeTopK)} />
                <button type="button" aria-label="增加 Top-K" disabled={busy || topK >= 50} onClick={() => setTopK(value => normalizeTopK(normalizeTopK(value) + 1))}><Plus size={14} strokeWidth={1.5} /></button>
              </div></div>
              </fieldset>
              <button className="primary query-button" disabled={unavailable || !query.trim()} type="submit">{busy ? <LoaderCircle className="spin" size={17} /> : <Search size={17} />}{busy ? '检索与生成中 · ' + elapsed + ' s' : '开始查询'}{!busy && <span>↵</span>}</button>
            </div>
            <div className="query-hint"><Settings2 size={14} /><span aria-live="polite">{methodId === 'lightrag' ? <>{modeDescriptions[options.mode] || method?.description}<small>{options.mode === 'naive' ? 'Top-K 控制候选文本片段数量。' : 'Top-K 控制检索候选数量，不等于最终展示的实体或关系总数。'}</small></> : method?.description || '正在加载检索方法…'}</span><kbd>Ctrl + Enter</kbd></div>
          </form>
          <div className={'results-grid' + (graphExpanded ? ' graph-expanded' : '')}>
            <section className="answer-panel" hidden={graphExpanded}><div className="section-heading"><div><span className="eyebrow">GROUNDED ANSWER</span><h2>回答与依据</h2></div></div>
              {busy ? <div className="empty-copy"><LoaderCircle className="spin" size={28} /><h3>正在检索证据并生成回答</h3><p>模型请求可能需要几分钟。请勿重复提交。<br />这里只显示实际等待时间，不模拟进度。</p><strong>{elapsed} s</strong></div> : result ? <>
                <div className="run-meta"><span>{result.method_id} / {modeLabel(result.method_id, result.options.mode || 'default')}</span><span><Clock3 size={13} />{seconds(result.latency_ms)}</span></div>
                <p className="result-provenance">{new Date(result.created_at).toLocaleString()} · {String(result.retrieval.metadata?.corpus_name || '数据来源未记录')}</p>
                {result.error && <div className="alert error" role="alert">{result.error.message}<small>{result.error.code}</small></div>}
                <div className="answer-content"><Markdown>{result.answer || '答案尚未生成。可在下方查看已获得的检索结果。'}</Markdown></div>
                <div className="timings"><span>检索 <b>{seconds(result.retrieval_ms)}</b></span><span>生成 <b>{seconds(result.generation_ms)}</b></span></div>
                {evidence?.warnings.map(w => <p className="warning-note" key={w}>{w}</p>)}
              </> : <div className="empty-copy answer-empty"><span className="empty-symbol"><BookOpen size={27} strokeWidth={1.3} /></span><h3>从一个好问题开始</h3><p>答案将基于实际检索到的证据生成。<br />你可以查看实体、关系与原文片段。</p><span className="empty-caption">YOUR QUESTION, CONNECTED.</span></div>}
            </section>
            <GraphPanel graph={graph} queried={!!result} expanded={graphExpanded} onToggleExpanded={() => setGraphExpanded(value => !value)} />
          </div>
          <section className="evidence-panel"><div className="section-heading"><div><span className="eyebrow">RETRIEVAL DETAILS</span><h2>回到原始证据</h2></div><span className="muted">只展示方法实际返回的数据</span></div>
            <div className="evidence-tabs" role="tablist" aria-label="检索结果类型">{tabs.map(([key, name]) => <button role="tab" id={'tab-' + key} aria-controls="evidence-content" aria-selected={tab === key} key={key} className={tab === key ? 'selected' : ''} onClick={() => setTab(key)}>{name}{key !== 'context' && <span>{evidence?.[key].length || 0}</span>}</button>)}</div>
            <div id="evidence-content" role="tabpanel" aria-labelledby={'tab-' + tab} className="evidence-content">
              {!evidence ? <p className="muted">完成一次查询后，这里会展示对应的检索证据。</p> : tab === 'context' ? <><p className="muted">按固定模板序列化的展示上下文；不是 LightRAG 内部最终 prompt。</p><pre>{evidence.context_text || '本次没有返回上下文。'}</pre></> :
                <EvidenceList records={evidence[tab]} kind={tab} />}
            </div>
          </section>
        </>}
        <footer><span>知图 / Retrieval Workbench</span><span>检索方法独立 · 答案生成统一 · 证据可追溯</span></footer>
      </div>
    </main>
    {apiSettingsOpen && <ApiSettings onClose={() => setApiSettingsOpen(false)} onSaved={() => { void refresh(true) }} />}
    <dialog ref={dialog} onCancel={() => setSetupOpen(false)} onClose={() => setSetupOpen(false)}>
      <div className="modal-heading"><h2>数据与索引</h2><button aria-label="关闭数据管理" onClick={() => setSetupOpen(false)}><X size={20} /></button></div>
      <p className="muted">一次选用一份文本、一个 workspace。仅选择文本不会产生 API 费用。</p>
      {error && <div className="alert error" role="alert">{error}</div>}
      <label className="field-label" htmlFor="subset">数据来源 · 自动选择最短文本</label><select id="subset" value={subset} onChange={e => setSubset(e.target.value)} disabled={settingUp || index.status === 'building'}><option value="novel">GraphRAG-Bench · Novel</option><option value="medical">GraphRAG-Bench · Medical</option></select>
      <button className="secondary wide" disabled={settingUp || busy || index.status === 'building'} onClick={() => void selectDataset()}>选择这份文本</button>
      <div className="setup-summary"><strong>{dataset?.corpus_name || '尚未选择'}</strong><span>{indexLabels[index.status]}{index.status === 'ready' && !index.reusable ? '（配置不匹配）' : ''}</span>{index.error_type && <p className="error-text">{index.error_type}</p>}</div>
      <p className="warning-note">建图会调用 LLM 抽取实体与关系，产生 API 费用；打开网页、选择文本、查看图谱不会重新建图。重建会保留原 workspace 备份。</p>
      <div className="modal-actions"><button className="secondary" disabled={!dataset || settingUp || busy || index.status === 'building'} onClick={() => void build(true)}>备份并重建</button><button className="primary" disabled={!dataset || settingUp || busy || index.status === 'building' || index.reusable} onClick={() => void build(false)}>{index.reusable ? '已有索引可复用' : '建立索引'}</button></div>
    </dialog>
  </div>
}

function EvidenceList({ records, kind }: { records: RecordData[]; kind: string }) {
  if (!records.length) return <p className="muted">本次没有返回此类数据。空列表不代表接口故障。</p>
  const label = tabs.find(([key]) => key === kind)?.[1] || '证据'
  return <div className="evidence-table-scroll"><table className="evidence-table" aria-label={label + '检索结果'}>
    <colgroup><col className="row-number-col" /><col /><col className="source-col" /></colgroup>
    <thead><tr><th scope="col">#</th><th scope="col">{label} / 内容</th><th scope="col">来源</th></tr></thead>
    <tbody>{records.map((record, i) => <tr key={i}>
      <td className="row-number">{String(i + 1).padStart(2, '0')}</td>
      <td>
        <div className="evidence-title">{String(record.entity_name || record.chunk_id || (record.src_id ? record.src_id + ' ↔ ' + record.tgt_id : '') || record.reference_id || '记录 ' + (i + 1))}</div>
        {(record.content != null || record.description != null) && <p className="evidence-preview">{textValue(record.content ?? record.description)}</p>}
        <details className="raw-fields">
          <summary>查看原始字段</summary>
          <dl>{Object.entries(record).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{textValue(value)}</dd></div>)}</dl>
        </details>
      </td>
      <td className="source-value">{record.file_path != null ? textValue(record.file_path) : '—'}</td>
    </tr>)}</tbody>
  </table></div>
}
