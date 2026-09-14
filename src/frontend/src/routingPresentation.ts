export const routingMethods = ['vector', 'lightrag', 'pathrag'] as const
export type RoutingMethod = typeof routingMethods[number]
export type RoutingDecision = { selected: RoutingMethod; scores: Record<RoutingMethod, number | null> }

export function readRoutingDecision(metadata?: Record<string, unknown>): RoutingDecision | null {
  const selected = metadata?.selected_method
  if (!routingMethods.includes(selected as RoutingMethod)) return null
  const values = metadata?.routing_probabilities
  const probabilities = values && typeof values === 'object' ? values as Record<string, unknown> : {}
  const scores = Object.fromEntries(routingMethods.map(method => {
    const value = probabilities[method]
    return [method, typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1 ? value : null]
  })) as RoutingDecision['scores']
  return { selected: selected as RoutingMethod, scores }
}
