import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mergeGraphHits, motionCandidates } from '../src/graphDisplay.ts'

test('full graph survives bounded hits, undirected pairs match, cache is unchanged', () => {
  const full = { nodes: ['a','b','c'].map(id => ({ id, retrieved: false })), edges: [
    { id: 'ab', source: 'a', target: 'b', directed: false },
    { id: 'bc', source: 'b', target: 'c', directed: true },
  ], total_nodes: 3, total_edges: 2, warnings: [], layout_key: 'stable' }
  const hits = { nodes: [], edges: [], hit_node_ids: ['c'], hit_edge_pairs: [['b','a'],['c','b']], warnings: [] }
  const result = mergeGraphHits(full, hits)
  assert.equal(result.nodes.length, 3)
  assert.equal(result.edges.length, 2)
  assert.equal(result.layout_key, 'stable')
  assert.equal(result.nodes[2].retrieved, true)
  assert.equal(result.edges[0].retrieved, true)
  assert.equal(result.edges[1].retrieved, false)
  assert.equal(full.nodes[2].retrieved, false)
})

test('older records still highlight recorded hits', () => {
  const full = { nodes: [{ id: 'a' }], edges: [], warnings: [] }
  assert.equal(mergeGraphHits(full, { nodes: [{id:'a',retrieved:true}], edges: [], warnings: [] }).nodes[0].retrieved, true)
})

test('old graph IDs report missing highlights without fuzzy matching', () => {
  const full = { nodes:[{id:'Alpha'}], edges:[], warnings:[] }
  const hits = { nodes:[], edges:[], hit_node_ids:['"ALPHA"'], hit_edge_pairs:[], warnings:[] }
  const result = mergeGraphHits(full,hits)
  assert.equal(result.nodes[0].retrieved,false)
  assert.ok(result.warnings.some(w=>w.includes('无法匹配') && w.includes('重新查询')))
  assert.equal(full.warnings.length,0)
})

test('motion work is bounded by visible nodes and affected edges', () => {
  const candidates = Array.from({length: 4500}, (_,i) => ({ id: String(i), x: i * 100, y: 0, degree: i === 0 ? 1000 : 15 }))
  const selected = motionCandidates(candidates, {x1:0,y1:-10,x2:1000000,y2:10})
  assert.ok(selected.length <= 40)
  assert.ok(selected.reduce((sum,n) => sum+n.degree,0) <= 400)
  assert.ok(!selected.some(n => n.id === '0'))
  assert.equal(motionCandidates(candidates, {x1:-100,y1:20,x2:100,y2:30}).length, 0)
})
