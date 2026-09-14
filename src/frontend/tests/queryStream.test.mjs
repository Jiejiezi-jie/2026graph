import test from 'node:test'
import assert from 'node:assert/strict'
import { streamQuery } from '../src/api.ts'

test('routing and evidence are consumed while the final answer is still pending, including split UTF-8', async () => {
  const original = globalThis.fetch
  let controller
  globalThis.fetch = async () => new Response(new ReadableStream({start(c) { controller = c }}))
  try {
    const events = []
    let finished = false
    const query = streamQuery({}, event => events.push(event)).then(result => { finished = true; return result })
    await new Promise(resolve => setImmediate(resolve))
    const encoder = new TextEncoder()
    const line = encoder.encode(JSON.stringify({type:'routing',id:'one',metadata:{name:'中文',selected_method:'pathrag'}})+'\n')
    for (const byte of line) controller.enqueue(new Uint8Array([byte]))
    await new Promise(resolve => setImmediate(resolve))
    assert.equal(events[0].metadata.name, '中文')
    assert.equal(finished, false)
    controller.enqueue(encoder.encode(JSON.stringify({type:'retrieval',id:'one',retrieval:{chunks:['evidence']}})+'\n'))
    await new Promise(resolve => setImmediate(resolve))
    assert.equal(events[1].type, 'retrieval')
    assert.equal(finished, false)
    controller.enqueue(encoder.encode(JSON.stringify({type:'result',run:{id:'one',answer:'done'}})))
    controller.close()
    assert.equal((await query).answer, 'done')
  } finally { globalThis.fetch = original }
})

test('an interrupted stream preserves delivered stages and reports the missing final response', async () => {
  const original = globalThis.fetch
  globalThis.fetch = async () => new Response('{"type":"routing","id":"one","metadata":{}}\n')
  try {
    const events = []
    await assert.rejects(streamQuery({}, event => events.push(event)), /连接中断/)
    assert.equal(events.length, 1)
  } finally { globalThis.fetch = original }
})

test('stream and HTTP errors reach the user instead of leaving loading active', async () => {
  const original = globalThis.fetch
  try {
    globalThis.fetch = async () => new Response('{"type":"error","error":{"message":"模型不可用"}}\n')
    await assert.rejects(streamQuery({}, () => {}), /模型不可用/)
    globalThis.fetch = async () => new Response('{"error":{"message":"工作区忙"}}', {status:409})
    await assert.rejects(streamQuery({}, () => {}), /工作区忙/)
  } finally { globalThis.fetch = original }
})
