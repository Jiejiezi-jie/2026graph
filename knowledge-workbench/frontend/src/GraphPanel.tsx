import { useEffect, useRef, useState } from 'react'
import cytoscape, { type Core } from 'cytoscape'
import { Focus, Minus, Plus, Network, Pause, Play, X, Maximize2, Minimize2 } from 'lucide-react'
import type { GraphData, GraphNode } from './api'
import { startGraphMotion } from './graphMotion'
import NodeDetails from './NodeDetails'
import { nodeSymbol } from './nodeSymbols'

const colors = ['#91ad9b', '#c3b18d', '#99abc0', '#b4a0b5', '#91adb0', '#adb395']
const borders = ['#6e8f7a', '#a3906a', '#778fa9', '#957e97', '#6e9295', '#8c956f']

export default function GraphPanel({ graph, queried, expanded, onToggleExpanded }: {
  graph: GraphData; queried: boolean; expanded: boolean; onToggleExpanded: () => void
}) {
  const canvas = useRef<HTMLDivElement>(null)
  const cy = useRef<Core | null>(null)
  const [selected, setSelected] = useState<GraphNode | null>(null)
  const [edgeInfo, setEdgeInfo] = useState<{ label: string; description: string } | null>(null)
  const [onlyHits, setOnlyHits] = useState(false)
  const [motionPaused, setMotionPaused] = useState(false)
  const paused = useRef(false)
  const expandedRef = useRef(expanded)
  expandedRef.current = expanded
  const resizeGraph = useRef<() => void>(() => {})
  const selectNode = useRef<(id: string) => void>(() => {})
  const types = [...new Set(graph.nodes.map(n => n.type))].sort()
  const visibleNodes = graph.nodes.filter(n => !onlyHits || n.retrieved || !queried)
  const visibleIds = new Set(visibleNodes.map(n => n.id))
  const visibleEdges = graph.edges.filter(e => visibleIds.has(e.source) && visibleIds.has(e.target) && (!onlyHits || e.retrieved || !queried))
  useEffect(() => {
    setSelected(null); setEdgeInfo(null)
    if (!canvas.current || !graph.nodes.length) return
    const palette = [...new Set(graph.nodes.map(n => n.type))].sort()
    const nodes = graph.nodes.filter(n => !onlyHits || n.retrieved || !queried)
    const ids = new Set(nodes.map(n => n.id))
    const instance = cytoscape({
      container: canvas.current,
      elements: [
        ...nodes.map((n, i) => ({ data: { ...n, id: 'node:' + n.id, originalId: n.id, color: colors[palette.indexOf(n.type) % colors.length], border: borders[palette.indexOf(n.type) % borders.length], symbol: nodeSymbol(n.type) },
          position: { x: Math.cos(i * 2.39996) * 75 * Math.sqrt(i), y: Math.sin(i * 2.39996) * 75 * Math.sqrt(i) },
          classes: n.retrieved && queried ? 'hit' : queried ? 'neighbor' : '' })),
        ...graph.edges.filter(e => ids.has(e.source) && ids.has(e.target) && (!onlyHits || e.retrieved || !queried))
          .map(e => ({ data: { ...e, id: 'link:' + e.id, source: 'node:' + e.source, target: 'node:' + e.target },
            classes: [e.retrieved && queried ? 'hit' : queried ? 'neighbor' : '', e.directed ? 'directed' : ''].join(' ') })),
      ],
      style: [
        { selector: 'node', style: {
          'background-color': 'data(color)', width: 38, height: 38, label: '',
          'background-image': 'data(symbol)', 'background-width': '52%', 'background-height': '52%',
          'background-fit': 'none', 'background-image-opacity': 1,
          'font-size': 12, 'font-family': 'Segoe UI, Microsoft YaHei, sans-serif', color: '#394742',
          'text-valign': 'bottom', 'text-margin-y': 9, 'text-max-width': '110px', 'text-wrap': 'ellipsis',
          'min-zoomed-font-size': 9,
          'border-width': 1, 'border-color': 'data(border)', 'overlay-opacity': 0,
        } },
        { selector: 'node.label-visible', style: { label: 'data(label)', 'min-zoomed-font-size': 0 } },
        { selector: 'edge', style: { width: .8, opacity: .55, 'line-color': '#bac9c0', 'curve-style': 'bezier', 'target-arrow-color': '#9bafa1', 'arrow-scale': .65 } },
        { selector: '.directed', style: { 'target-arrow-shape': 'triangle' } },
        { selector: 'node.hit', style: { 'border-width': 2, 'border-color': '#448364', 'font-weight': 'bold' } },
        { selector: 'edge.hit', style: { width: 1.8, opacity: .85, 'line-color': '#66927a', 'target-arrow-color': '#66927a' } },
        { selector: 'node.neighbor', style: { opacity: .7 } },
        { selector: 'node.adjacent', style: { opacity: 1 } },
        { selector: 'edge.adjacent, edge:selected', style: { width: 1.8, opacity: .95, 'line-color': '#648e78', 'target-arrow-color': '#648e78' } },
        { selector: 'node.dimmed', style: { opacity: .22, 'text-opacity': 0 } },
        { selector: 'node.hit.dimmed', style: { opacity: .6 } },
        { selector: 'edge.dimmed', style: { opacity: .12 } },
        { selector: 'node.hovered, node:selected', style: { label: 'data(label)', 'text-wrap': 'wrap', 'text-max-width': '220px', 'text-opacity': 1, 'min-zoomed-font-size': 0, opacity: 1, 'z-index': 10 } },
        { selector: 'node:selected', style: { 'border-width': 2, 'border-color': '#448364', 'underlay-color': '#76a58b', 'underlay-opacity': .16, 'underlay-padding': 7, 'font-weight': 'bold', color: '#2d6348' } },
      ],
      layout: { name: 'cose', animate: false, fit: false, randomize: false,
        nodeRepulsion: () => 18000, idealEdgeLength: () => 160, gravity: .15,
        componentSpacing: 160, nodeDimensionsIncludeLabels: true, padding: 45 },
      minZoom: 0.03, maxZoom: 3, selectionType: 'single', boxSelectionEnabled: false,
    })
    cy.current = instance
    // Expand the settled layout without enlarging the nodes or their labels.
    instance.nodes().positions(node => {
      const p = node.position()
      return { x: p.x * 1.85, y: p.y * 1.85 }
    })
    // Start at reading scale around a connected node; the rest remains offscreen.
    const anchors = queried && instance.nodes('.hit').length ? instance.nodes('.hit') : instance.nodes()
    const anchor = anchors.sort((a, b) => b.degree(false) - a.degree(false)).first()
    instance.nodes().sort((a, b) => b.degree(false) - a.degree(false)).slice(0, 6).addClass('core')
    instance.zoom(1)
    if (anchor.length) instance.center(anchor)
    if (expandedRef.current) instance.fit(undefined, 48)
    function updateLabels() {
      const zoom = instance.zoom()
      const fontSize = Math.min(60, Math.max(12, 10 / zoom))
      const maxWidth = Math.max(110, 90 / zoom)
      instance.batch(() => {
        // Keep important labels readable and reserve their screen space first.
        instance.nodes().style({ width: Math.min(90, Math.max(38, 24 / zoom)), height: Math.min(90, Math.max(38, 24 / zoom)), 'font-size': fontSize, 'text-max-width': maxWidth + 'px' })
        instance.nodes().removeClass('label-visible')
        const occupied: { x: number; y: number; width: number; height: number }[] = []
        const priority = (node: cytoscape.NodeSingular) => node.selected() ? 10000 : node.hasClass('hovered') ? 9000 : node.hasClass('hit') ? 8000 : node.hasClass('core') ? 7000 : node.hasClass('adjacent') ? 6000 : node.degree(false)
        instance.nodes().sort((a, b) => priority(b) - priority(a)).forEach(node => {
          const forced = node.selected() || node.hasClass('hovered')
          if (!forced && (node.hasClass('dimmed') || (zoom < 1.3 && !node.hasClass('core') && !node.hasClass('hit') && !node.hasClass('adjacent')))) return
          const p = node.renderedPosition()
          if (!forced && (p.x < 0 || p.x > instance.width() || p.y < 0 || p.y > instance.height())) return
          const width = Math.min(String(node.data('label')).length * fontSize * .6, maxWidth) * zoom + 16
          const box = { x: p.x - width / 2, y: p.y + node.renderedHeight() / 2 + 9 * zoom, width, height: fontSize * zoom * 1.4 + 8 }
          if (forced || !occupied.some(other => box.x < other.x + other.width && box.x + box.width > other.x && box.y < other.y + other.height && box.y + box.height > other.y)) {
            node.addClass('label-visible'); occupied.push(box)
          }
        })
      })
    }
    updateLabels()
    let labelFrame = 0
    function scheduleLabels() {
      cancelAnimationFrame(labelFrame)
      labelFrame = requestAnimationFrame(updateLabels)
    }
    instance.on('zoom pan', scheduleLabels)
    let focusedNodeId: string | null = null
    function clearFocus() {
      focusedNodeId = null
      instance.elements().unselect().removeClass('adjacent dimmed')
      setSelected(null); setEdgeInfo(null)
      scheduleLabels()
    }
    selectNode.current = id => {
      if (!id || id === focusedNodeId) {
        instance.stop()
        clearFocus()
        return
      }
      const element = instance.getElementById('node:' + id)
      clearFocus()
      if (!element.length) return
      focusedNodeId = id
      element.select()
      instance.elements().addClass('dimmed')
      element.closedNeighborhood().removeClass('dimmed').addClass('adjacent')
      setSelected(graph.nodes.find(n => n.id === id) || null)
      const zoom = Math.max(instance.zoom(), 1.2)
      // The detail is an overlay: move the focus into the unobscured area.
      const sideWidth = expandedRef.current && instance.width() >= 720 ? 368 : 0
      const bottomHeight = instance.width() < 720 ? expandedRef.current ? Math.min(260, instance.height() * .44) : 242 : 0
      const position = element.position()
      instance.stop()
      instance.animate({ zoom, pan: { x: (instance.width() - sideWidth) / 2 - position.x * zoom, y: (instance.height() - bottomHeight) / 2 - position.y * zoom } },
        { duration: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 320 })
    }
    instance.on('tap', 'node', event => selectNode.current(event.target.data('originalId')))
    // Cytoscape applies native selection after tap; keep a dismissed node dismissed.
    instance.on('tapselect', 'node', event => {
      if (event.target.data('originalId') !== focusedNodeId) event.target.unselect()
    })
    instance.on('mouseover', 'node', event => { event.target.addClass('hovered'); scheduleLabels() })
    instance.on('mouseout', 'node', event => { event.target.removeClass('hovered'); scheduleLabels() })
    instance.on('tap', 'edge', event => {
      clearFocus()
      event.target.select()
      instance.elements().addClass('dimmed')
      event.target.connectedNodes().add(event.target).removeClass('dimmed').addClass('adjacent')
      setEdgeInfo({ label: event.target.data('label'), description: event.target.data('description') }); setSelected(null)
    })
    instance.on('tap', event => {
      if (event.target === instance) selectNode.current('')
    })
    let width = instance.width(), height = instance.height()
    let lastExpanded = expandedRef.current
    let splitView: { zoom: number; center: { x: number; y: number } } | null = null
    resizeGraph.current = () => {
      const changedMode = lastExpanded !== expandedRef.current
      if (changedMode) {
        instance.stop()
        if (expandedRef.current) splitView = { zoom: instance.zoom(), center: { x: (width / 2 - instance.pan().x) / instance.zoom(), y: (height / 2 - instance.pan().y) / instance.zoom() } }
      }
      instance.resize()
      const nextWidth = instance.width(), nextHeight = instance.height()
      if (changedMode) {
        instance.fit(undefined, 48)
        if (!expandedRef.current && splitView) {
          instance.viewport({ zoom: splitView.zoom, pan: { x: nextWidth / 2 - splitView.center.x * splitView.zoom, y: nextHeight / 2 - splitView.center.y * splitView.zoom } })
        }
        lastExpanded = expandedRef.current
      } else instance.panBy({ x: (nextWidth - width) / 2, y: (nextHeight - height) / 2 })
      width = nextWidth; height = nextHeight
    }
    const observer = new ResizeObserver(() => resizeGraph.current())
    observer.observe(canvas.current)
    const stopMotion = startGraphMotion(instance, canvas.current, () => paused.current)
    return () => { stopMotion(); cancelAnimationFrame(labelFrame); observer.disconnect(); resizeGraph.current = () => {}; selectNode.current = () => {}; instance.destroy(); cy.current = null }
  }, [graph, onlyHits, queried])

  useEffect(() => {
    const frame = requestAnimationFrame(() => resizeGraph.current())
    return () => cancelAnimationFrame(frame)
  }, [expanded])

  function focusNode(id: string) {
    selectNode.current(id)
  }

  function zoomBy(factor: number) {
    const instance = cy.current
    if (instance) instance.zoom({ level: instance.zoom() * factor, renderedPosition: { x: instance.width() / 2, y: instance.height() / 2 } })
  }

  return <section className={'graph-panel' + (expanded ? ' is-expanded' : '')} aria-label="知识图谱工作区">
    <div className="section-heading"><div><span className="eyebrow">KNOWLEDGE GRAPH</span>
      <h2>{queried ? '本次查询相关图谱' : '已建图谱预览'}</h2></div>
      <div className="graph-heading-actions"><span className="graph-count">{visibleNodes.length} 节点 · {visibleEdges.length} 关系</span>
        <button className="graph-expand-button" aria-expanded={expanded} aria-controls="knowledge-graph-stage" onClick={onToggleExpanded}>{expanded ? <Minimize2 size={15} /> : <Maximize2 size={15} />}{expanded ? '收起图谱' : '展开图谱'}</button>
      </div>
    </div>
    <div className="graph-tools">
      <label className="sr-only" htmlFor="node-search">定位图中实体</label>
      <select id="node-search" value={selected?.id || ''} onChange={e => focusNode(e.target.value)}>
        <option value="">定位图中实体…</option>
        {visibleNodes.map(n => <option key={n.id} value={n.id}>{n.label}</option>)}
      </select>
      {queried && <label className="checkbox-label"><input type="checkbox" checked={onlyHits} onChange={e => setOnlyHits(e.target.checked)} />仅命中</label>}
      <div className="zoom-tools">
        <button aria-label={motionPaused ? '恢复漂动' : '暂停漂动'} title={motionPaused ? '恢复漂动' : '暂停漂动'} aria-pressed={motionPaused} onClick={() => { paused.current = !paused.current; setMotionPaused(paused.current) }}>{motionPaused ? <Play size={15} /> : <Pause size={15} />}</button>
        <button aria-label="放大图谱" onClick={() => zoomBy(1.2)}><Plus size={16} /></button>
        <button aria-label="缩小图谱" onClick={() => zoomBy(1 / 1.2)}><Minus size={16} /></button>
        <button aria-label="适应画布" title="取消聚焦并查看全部已加载节点" onClick={() => { focusNode(''); cy.current?.fit(undefined, 48) }}><Focus size={16} /></button>
      </div>
    </div>
    <div className="graph-stage" id="knowledge-graph-stage">
      <div ref={canvas} className="graph-canvas" role="img" aria-label={queried ? '绿色描边为实际检索命中，淡色为展示用邻居。可用上方实体列表定位。' : '当前索引的有限节点预览，可拖动节点和缩放。'} />
      {!graph.nodes.length && <div className="graph-empty"><Network size={44} strokeWidth={1} /><h3>{queried ? '本次没有可展示的图节点' : '图谱将在这里展开'}</h3><p>{queried ? '纯文本方法可以只返回 chunks，不必提供图谱。' : '建立或载入索引后，查看实体之间的连接。'}</p></div>}
      {selected && <NodeDetails key={selected.id} node={selected} nodes={visibleNodes} edges={visibleEdges} queried={queried}
        colorForType={type => colors[types.indexOf(type) % colors.length]} onSelect={focusNode} onClose={() => focusNode('')} />}
      {edgeInfo && <aside className="node-detail" aria-label="关系详情">
        <button className="close-icon" aria-label="关闭实体详情" onClick={() => focusNode('')}><X size={15} /></button>
        <span className="eyebrow">RELATIONSHIP</span>
        <h3>{edgeInfo.label || '关系详情'}</h3>
        <p>{edgeInfo.description || '索引中未提供描述。'}</p>
      </aside>}
    </div>
    <p className="graph-navigation-hint">点击节点聚焦 · 再次点击取消高亮，保持视角 · 拖动画布探索 · 滚轮缩放</p>
    <div className="graph-legend">{types.map((type, index) =>
      <span key={type}><i style={{ background: colors[index % colors.length] }}><img src={nodeSymbol(type)} alt="" /></i>{type}</span>)}
      {!types.length && <span>实体类型由真实索引提供</span>}
    </div>
    <p className="graph-note">{queried ? '绿色描边 / 连线：实际命中；淡色：展示补充的一跳邻居。不是模型推理过程。' : '按连接数选取最多 60 个节点预览；不是全图，也不是检索结果。'}
      {graph.truncated && ' 已限制展示规模。'}</p>
    {graph.warnings.map(w => <p key={w} className="graph-note">{w}</p>)}
  </section>
}
