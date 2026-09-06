export type RecordData = Record<string, unknown>
export type GraphNode = { id: string; label: string; type: string; description: string; retrieved: boolean }
export type GraphEdge = { id: string; source: string; target: string; label: string; description: string; retrieved: boolean; directed: boolean }
export type GraphData = { nodes: GraphNode[]; edges: GraphEdge[]; total_nodes: number; total_edges: number; truncated: boolean; warnings: string[] }
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
export type Health = { api_key_configured: boolean; llm_model: string; embedding_model: string }
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
