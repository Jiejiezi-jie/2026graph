import type { GraphData } from './api'

/** A bounded record supplies highlights, never replaces the complete display graph. */
export function mergeGraphHits(full: GraphData, hits?: GraphData): GraphData {
  if (!hits) return full
  const ids = new Set(hits.hit_node_ids?.length ? hits.hit_node_ids : hits.nodes.filter(n => n.retrieved).map(n => n.id))
  const pairs = new Set((hits.hit_edge_pairs?.length ? hits.hit_edge_pairs : hits.edges.filter(e => e.retrieved).map(e => [e.source, e.target])).map(pair => JSON.stringify(pair)))
  return { ...full,
    nodes: full.nodes.map(n => ({ ...n, retrieved: ids.has(n.id) })),
    edges: full.edges.map(e => ({ ...e, retrieved: pairs.has(JSON.stringify([e.source, e.target])) || (!e.directed && pairs.has(JSON.stringify([e.target, e.source]))) })),
    warnings: [...new Set([...full.warnings, ...hits.warnings])],
  }
}

type MotionCandidate = { id: string; x: number; y: number; degree: number }
export function motionCandidates<T extends MotionCandidate>(nodes: T[], extent: {x1:number;y1:number;x2:number;y2:number}): T[] {
  const chosen: T[] = []
  let edges = 0
  for (const node of nodes) {
    if (node.x < extent.x1 || node.x > extent.x2 || node.y < extent.y1 || node.y > extent.y2 || node.degree > 40 || edges + node.degree > 400) continue
    chosen.push(node); edges += node.degree
    if (chosen.length === 40) break
  }
  return chosen
}
