async (page) => {
  // Read-only graph UI checks. Never submits a query or writes to the backend.
  const assert = (value, message) => { if (!value) throw new Error(message) }
  const state = () => page.evaluate(() => {
    const cy = document.querySelector('.graph-canvas')._cyreg.cy;
    const nodes = cy.nodes().map(n => ({ id: n.id(), ...n.position() }));
    const visible = cy.nodes().filter(n => {
      const p = n.renderedPosition();
      return p.x >= 0 && p.y >= 0 && p.x <= cy.width() && p.y <= cy.height();
    }).length;
    return { nodes, visible, zoom: cy.zoom(), pan: cy.pan() };
  });
  const distance = (a, b) => Math.max(...a.nodes.map((n, i) => Math.hypot(n.x - b.nodes[i].x, n.y - b.nodes[i].y)));
  try {
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    await page.setViewportSize({ width: 1440, height: 1050 });
    await page.reload();
    await page.waitForFunction(() => document.querySelector('.graph-canvas')?._cyreg?.cy?.nodes().length > 10);
    await page.locator('.graph-stage').scrollIntoViewIfNeeded();
    await page.mouse.move(10, 10);
    const initial = await state();
    assert(initial.visible > 0 && initial.visible < initial.nodes.length, 'Initial view must show a readable local region, not fit every node');
    await page.waitForTimeout(1200);
    const drifting = await state();
    assert(distance(initial, drifting) > 0.5 && distance(initial, drifting) < 20, 'Idle motion must be gentle and bounded');
    const box = await page.locator('.graph-canvas').boundingBox();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    const hover = await state();
    await page.waitForTimeout(400);
    assert(distance(hover, await state()) > 0.2, 'Hovering the canvas must keep the gentle motion running');
    const entityId = await page.locator('#node-search option').nth(1).getAttribute('value');
    await page.getByRole('combobox', { name: '定位图中实体' }).selectOption(entityId);
    const selected = await state();
    await page.waitForTimeout(400);
    assert(distance(selected, await state()) > 0.2, 'Viewing entity details must not pause the graph');
    await page.mouse.move(10, 10);
    await page.getByRole('button', { name: '暂停漂动', exact: true }).click();
    const paused = await state();
    await page.waitForTimeout(400);
    assert(distance(paused, await state()) < 0.01, 'Pause must hold node positions');
    await page.getByRole('button', { name: '放大图谱', exact: true }).click();
    const zoomed = await state();
    assert(zoomed.zoom > paused.zoom, 'Zoom in must increase the local view scale');
    await page.setViewportSize({ width: 1200, height: 1000 });
    await page.waitForTimeout(150);
    assert(Math.abs((await state()).zoom - zoomed.zoom) < 0.001, 'Resize must preserve user zoom instead of fitting the graph again');
    await page.getByRole('button', { name: '适应画布', exact: true }).click();
    const fitted = await state();
    assert(fitted.visible === fitted.nodes.length, 'Fit control must still reveal all loaded nodes');
    await page.getByRole('button', { name: '恢复漂动', exact: true }).click();
    const resumed = await state();
    await page.waitForTimeout(400);
    assert(distance(resumed, await state()) > 0.2, 'Resume must restart motion even while an entity remains selected');
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.waitForTimeout(100);
    const reduced = await state();
    await page.waitForTimeout(400);
    assert(distance(reduced, await state()) < 0.01, 'Reduced-motion preference must stop canvas animation');
    return { initialVisible: initial.visible, loaded: initial.nodes.length, initialZoom: initial.zoom, idleMovement: distance(initial, drifting), checks: ['local view', 'idle drift', 'hover keeps drifting', 'selection keeps drifting', 'pause and resume buttons', 'zoom', 'resize preserves zoom', 'fit overview', 'reduced motion'] };
  } finally {
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    await page.setViewportSize({ width: 1440, height: 1050 });
    await page.reload();
  }
}
