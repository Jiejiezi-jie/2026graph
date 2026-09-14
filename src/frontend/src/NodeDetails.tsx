import { useId, useState } from 'react'
import { ChevronDown, X } from 'lucide-react'
import type { GraphNode, GraphEdge } from './api'
import { nodeSymbol } from './nodeSymbols'

const typeNames: Record<string, string> = { person: '人物', location: '地点', organization: '组织', event: '事件', concept: '概念', artifact: '物件', work: '作品', other: '其他', unknown: '未分类' }
const typeName = (type: string) => typeNames[type.toLowerCase()] || type

export default function NodeDetails({ node, nodes, edges, queried, colorForType, onSelect, onClose }: {
  node: GraphNode; nodes: GraphNode[]; edges: GraphEdge[]; queried: boolean
  colorForType: (type: string) => string; onSelect: (id: string) => void; onClose: () => void
}) {
  const [descriptionOpen, setDescriptionOpen] = useState(false)
  const descriptionId = useId()
  const relatedEdges = edges.filter(edge => edge.source === node.id || edge.target === node.id)
  const relatedIds = new Set(relatedEdges.flatMap(edge => [edge.source, edge.target]))
  const neighbors = nodes.filter(item => item.id !== node.id && relatedIds.has(item.id))
  const longDescription = node.description.length > 240

  return <aside className="node-detail entity-detail" aria-label="节点详情">
    <header className="entity-detail-header">
      <span className="entity-avatar" style={{ background: colorForType(node.type) }} aria-hidden="true"><img src={nodeSymbol(node.type)} alt="" /></span>
      <div><h3>{node.label}</h3><span className="entity-type-badge">{typeName(node.type)}</span></div>
      <button className="close-icon" aria-label="关闭实体详情" onClick={onClose}><X size={16} /></button>
    </header>
    <dl className="entity-facts">
      <div><dt>类型</dt><dd>{node.type}</dd></div>
      {node.id !== node.label && <div><dt>节点标识</dt><dd>{node.id}</dd></div>}
      <div><dt>关系数</dt><dd><strong>{relatedEdges.length}</strong><span className="fact-scope">当前图中</span></dd></div>
      {queried && <div><dt>检索状态</dt><dd className="graph-selection-status">{node.retrieved ? '检索命中' : '展示补充邻居'}</dd></div>}
    </dl>
    <section className="entity-description" aria-label="节点描述">
      <h4>节点描述</h4>
      <p id={descriptionId} className={'description-copy' + (!longDescription || descriptionOpen ? ' is-open' : '')}>{node.description || '索引中未提供描述。'}</p>
      {longDescription && <button className="description-toggle" aria-expanded={descriptionOpen} aria-controls={descriptionId} onClick={() => setDescriptionOpen(value => !value)}>{descriptionOpen ? '收起描述' : '展开完整描述'}<ChevronDown size={14} /></button>}
    </section>
    <details className="node-neighbors" open>
      <summary>相关实体（{neighbors.length}）<ChevronDown size={15} /></summary>
      <div className="neighbor-list">
        {neighbors.map(item => {
          const connections = relatedEdges.filter(edge => edge.source === item.id || edge.target === item.id)
          const relation = [...new Set(connections.map(edge => edge.label).filter(Boolean))].join(' · ')
          return <button key={item.id} data-node-id={item.id} onClick={() => onSelect(item.id)}>
            <i className="entity-dot" style={{ background: colorForType(item.type) }} aria-hidden="true"><img src={nodeSymbol(item.type)} alt="" /></i>
            <span className="neighbor-name">{item.label}{relation && <span className="neighbor-relation" title={relation}>{relation}</span>}</span>
            <small>{typeName(item.type)}</small>
          </button>
        })}
        {!neighbors.length && <p>当前图中没有相邻实体。</p>}
      </div>
    </details>
  </aside>
}
