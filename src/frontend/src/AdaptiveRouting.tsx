import { useEffect, useMemo, useState } from 'react'
import { Check, CircleAlert, FileText, LoaderCircle, Network, Route } from 'lucide-react'
import type { Run, RoutingUpdate } from './api'
import { readRoutingDecision, routingMethods } from './routingPresentation'
import './adaptiveRouting.css'

const candidates = {
  vector: { name: 'Vector', description: '文本向量 · 语义相似', icon: FileText },
  lightrag: { name: 'LightRAG', description: '图谱检索 · 实体与关系', icon: Network },
  pathrag: { name: 'PathRAG', description: '路径检索 · 多跳关联', icon: Route },
}

export default function AdaptiveRouting({ result, routing, busy, error, animate }: {
  result: Run | null; routing: RoutingUpdate | null; busy: boolean; error: string; animate: boolean
}) {
  const metadata = routing?.metadata ?? (result?.method_id === 'adaptive' ? result.retrieval.metadata : undefined)
  const routingId = routing?.id ?? result?.id
  const decision = useMemo(() => readRoutingDecision(metadata), [metadata])
  const [reducedMotion, setReducedMotion] = useState(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches)
  const [reveal, setReveal] = useState<{id: string; step: number; complete: boolean} | null>(null)
  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)')
    const update = () => setReducedMotion(media.matches)
    media.addEventListener('change', update)
    return () => media.removeEventListener('change', update)
  }, [])
  useEffect(() => {
    if (!decision || !routingId || !animate || reducedMotion) { setReveal(null); return }
    const id = routingId
    let active = true
    setReveal({id, step: 0, complete: false})
    const timers = [160, 380, 600].map((delay, i) => window.setTimeout(() => {
      if (active) setReveal({id, step: i + 1, complete: false})
    }, delay))
    timers.push(window.setTimeout(() => {
      if (active) setReveal({id, step: 3, complete: true})
    }, 960))
    return () => { active = false; timers.forEach(clearTimeout) }
  }, [decision, routingId, animate, reducedMotion])

  const showImmediately = !animate || reducedMotion
  const step = showImmediately ? 3 : reveal?.id === routingId ? reveal?.step ?? 0 : 0
  const complete = !!decision && (showImmediately || (reveal?.id === routingId && reveal?.complete))
  const waiting = busy && !decision
  const revealing = !!decision && !complete
  const failed = !decision && !busy && (!!error || !!result)
  const state = waiting ? 'waiting' : revealing ? 'revealing' : complete ? 'complete' : failed ? 'failed' : 'idle'
  const title = complete ? 'Adaptive 已完成路由' : waiting || revealing ? 'Adaptive 计算中'
    : failed ? 'Adaptive 路由结果不可用' : 'Adaptive 等待查询'
  const description = waiting ? '正在根据问题语义评估检索方式'
    : revealing ? '分类结果已返回，正在展示各项分数'
    : complete ? '已选择 ' + candidates[decision!.selected].name + ' 执行本次检索。'
    : failed ? '本次请求未返回分类结果，请检查错误提示后重试。'
    : '提出问题后，将展示三种检索方式的分类分数。'
  const StatusIcon = complete ? Check : failed ? CircleAlert : waiting || revealing ? LoaderCircle : Route
  return <section className="adaptive-routing" aria-label="Adaptive 路由评估" data-state={state} aria-busy={waiting || revealing}>
    <div className="routing-summary">
      <span className={'routing-status-icon' + (waiting || revealing ? ' is-loading' : '')}><StatusIcon size={21} strokeWidth={1.7} aria-hidden="true" /></span>
      <div><h3 role="status" aria-live="polite">{title}</h3><p>{description}</p></div>
    </div>
    <div className="routing-candidates" role="list" aria-label="候选检索方式">
      {routingMethods.map((method, index) => {
        const candidate = candidates[method]
        const Icon = candidate.icon
        const visible = !!decision && step > index
        const score = decision?.scores[method] ?? null
        const chosen = complete && decision?.selected === method
        const loading = waiting || (revealing && !visible)
        return <article key={method} role="listitem" className={'routing-candidate' + (chosen ? ' is-selected' : '')}
          data-method={method} data-score-visible={visible} aria-label={candidate.name + (chosen ? '，已选中' : '')}>
          <div className="routing-candidate-title"><span className="routing-method-icon"><Icon size={20} strokeWidth={1.6} aria-hidden="true" /></span><strong>{candidate.name}</strong>
            <span className="routing-selected-mark" aria-hidden={!chosen}>{chosen && <Check size={12} strokeWidth={2} />}</span></div>
          <div className="routing-score-row">
            <div className={'routing-score-track' + (loading ? ' is-loading' : '')} aria-hidden="true">
              <span className="routing-score-fill" style={{transform: 'scaleX(' + (visible && score !== null ? score : 0) + ')'}} />
            </div>
            <span className={'routing-score' + (visible ? ' is-visible' : '')} aria-label={loading ? '正在评估' : undefined}>
              {loading ? <span className="routing-score-skeleton" aria-hidden="true" /> : visible && score !== null ? (score * 100).toFixed(1) + '%' : '—'}
            </span>
          </div>
          <p>{candidate.description}</p>
        </article>
      })}
    </div>
    <p className="routing-footnote">{metadata?.router_kind === 'bge_m3_cls_logistic_regression' ? 'BGE-M3 语义路由 · ' : ''}分数为检索方法的分类概率，不代表答案正确率。{complete && decision && Object.values(decision.scores).some(score => score === null) ? ' 部分分类分数未返回。' : ''}</p>
  </section>
}
