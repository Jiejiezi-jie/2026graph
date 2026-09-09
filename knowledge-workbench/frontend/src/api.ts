export type RecordData = Record<string, unknown>
export type GraphNode = { id: string; label: string; type: string; description: string; retrieved: boolean; position?: { x: number; y: number } | null }
export type GraphEdge = { id: string; source: string; target: string; label: string; description: string; retrieved: boolean; directed: boolean }
export type GraphData = { nodes: GraphNode[]; edges: GraphEdge[]; total_nodes: number; total_edges: number; truncated: boolean; warnings: string[]; layout_key?: string | null; hit_node_ids?: string[]; hit_edge_pairs?: string[][] }
export type Retrieval = {
  context_text: string | null; entities: RecordData[]; relationships: RecordData[];
  chunks: RecordData[]; references: RecordData[]; graph: GraphData; warnings: string[]; metadata: RecordData
}
export type Run = {
  id: string; created_at: string; query: string; method_id: string; options: Record<string, string>;
  top_k: number; status: string; answer: string | null; latency_ms: number;
  retrieval_ms: number; generation_ms: number; retrieval: Retrieval;
  error: { code: string; message: string } | null
}
export type Method = { id: string; name: string; description: string; options: { key: string; label: string; choices: string[]; default: string }[] }
export type Dataset = { corpus_name: string; subset: string; character_count: number; word_count: number }
export type IndexStatus = { status: string; reusable: boolean; error_type: string | null }
export type Health = { api_key_configured: boolean; llm_model: string; embedding_model: string; imported_profile?: boolean; retrieval_ready?: boolean; runtime_issues?: string[] }
export const emptyGraph: GraphData = { nodes: [], edges: [], total_nodes: 0, total_edges: 0, truncated: false, warnings: [] }

export async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch('/api' + path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  })
  const text = await response.text()
  let data
  try { data = JSON.parse(text) } catch { throw new Error('后端未返回有效响应，请确认 FastAPI 已启动。') }
  if (!response.ok) throw new Error(data.error?.message || '请求参数无效或服务暂不可用。')
  return data as T
}

export type RoutingUpdate = { id: string; metadata: RecordData }
export type QueryEvent =
  | ({ type: 'routing' } & RoutingUpdate)
  | { type: 'retrieval'; id: string; retrieval: Retrieval }
  | { type: 'result'; run: Run }
  | { type: 'error'; error: { code: string; message: string } }
  | { type: 'heartbeat' }

export async function streamQuery(body: unknown, onEvent: (event: QueryEvent) => void): Promise<Run> {
  const response = await fetch('/api/query/stream', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  })
  if (!response.ok) {
    const data = await response.json().catch(() => null)
    throw new Error(data?.error?.message || '查询服务暂不可用，请确认后端已更新并启动。')
  }
  if (!response.body) throw new Error('浏览器未能读取查询数据流。')
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let result: Run | null = null
  const consume = (line: string) => {
    if (!line.trim()) return
    let event: QueryEvent
    try { event = JSON.parse(line) } catch { throw new Error('查询数据流格式错误，请重试。') }
    if (event.type === 'error') throw new Error(event.error.message)
    if (event.type === 'result') result = event.run
    onEvent(event)
  }
  try {
    while (true) {
      const {done, value} = await reader.read()
      buffer += decoder.decode(value, {stream: !done})
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      lines.forEach(consume)
      if (done) { consume(buffer); break }
    }
    if (!result) throw new Error('查询连接中断，未收到完整回答。已返回的路由和证据仍可查看。')
    return result
  } finally {
    await reader.cancel().catch(() => {})
    reader.releaseLock()
  }
}
