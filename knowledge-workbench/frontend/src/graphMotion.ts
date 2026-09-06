import type { Core } from 'cytoscape'

/** Small, reversible position offsets; graph data and viewport never change here. */
export function startGraphMotion(cy: Core, container: HTMLElement, paused: () => boolean) {
  const preference = window.matchMedia('(prefers-reduced-motion: reduce)')
  let visible = false
  let frame = 0
  let previous = performance.now()
  let elapsed = 0
  const nodes = cy.nodes()
  const visibility = new IntersectionObserver(entries => { visible = entries[0].isIntersecting })
  visibility.observe(container)

  function tick(now: number) {
    frame = requestAnimationFrame(tick)
    if (now - previous < 1000 / 30) return
    const delta = Math.min(now - previous, 64) / 1000
    previous = now
    if (!visible || document.hidden || preference.matches || paused()) return
    const next = elapsed + delta
    cy.batch(() => {
      nodes.forEach((node, i) => {
        if (node.grabbed() || node.locked()) return
        const phase = i * 2.39996
        const speed = .32 + (i % 5) * .035
        // Apply differences so pausing and dragging cannot cause position jumps.
        const dx = 7 * (Math.sin(next * speed + phase) - Math.sin(elapsed * speed + phase))
        const dy = 5 * (Math.cos(next * speed * .8 + phase) - Math.cos(elapsed * speed * .8 + phase))
        const p = node.position()
        node.position({ x: p.x + dx, y: p.y + dy })
      })
    })
    elapsed = next
  }
  frame = requestAnimationFrame(tick)
  return () => {
    cancelAnimationFrame(frame)
    visibility.disconnect()
  }
}
