import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readRoutingDecision } from '../src/routingPresentation.ts'

test('uses actual selected method and exact returned probabilities', () => {
  const decision = readRoutingDecision({
    selected_method: 'pathrag',
    routing_probabilities: {vector: .3721085562, lightrag: .1219613205, pathrag: .5059301233},
  })
  assert.equal(decision.selected, 'pathrag')
  assert.equal(decision.scores.vector, .3721085562)
  assert.equal(decision.scores.pathrag, .5059301233)
})

test('missing classification stays unavailable rather than inventing a winner', () => {
  assert.equal(readRoutingDecision(undefined), null)
  assert.equal(readRoutingDecision({routing_probabilities:{vector:1}}), null)
  assert.equal(readRoutingDecision({selected_method:'unknown'}), null)
})

test('absent or invalid scores are unknown, while real zero remains zero', () => {
  const decision = readRoutingDecision({selected_method:'vector',
    routing_probabilities:{vector:0, lightrag:NaN, pathrag:'0.8'}})
  assert.deepEqual(decision.scores, {vector:0,lightrag:null,pathrag:null})
  assert.equal(readRoutingDecision({selected_method:'pathrag', routing_probabilities:{pathrag:1.5}}).scores.pathrag, null)
})
