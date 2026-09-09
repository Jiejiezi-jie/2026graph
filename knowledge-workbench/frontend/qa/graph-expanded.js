async (page) => {
  // UI-only checks against the existing index; no model requests or data writes.
  const assert = (ok, message) => { if (!ok) throw new Error(message) };
  const state = () => page.evaluate(() => {
    const cy = document.querySelector('.graph-canvas')._cyreg.cy;
    const rect = selector => { const r = document.querySelector(selector).getBoundingClientRect(); return { x:r.x, width:r.width, height:r.height }; };
    return { canvas:rect('.graph-canvas'), panel:rect('.graph-panel'), grid:rect('.results-grid'),
      zoom:cy.zoom(), pan:cy.pan(), selected:cy.nodes(':selected').map(n=>n.id()),
      nodes:cy.nodes().map(n=>({id:n.id(),x:n.position('x'),y:n.position('y')})),
      visible:cy.nodes().filter(n=>{const p=n.renderedPosition();return p.x>=0&&p.y>=0&&p.x<=cy.width()&&p.y<=cy.height()}).length };
  });
  try {
    await page.setViewportSize({width:1440,height:1050});
    await page.reload();
    await page.waitForFunction(()=>document.querySelector('.graph-canvas')?._cyreg?.cy?.nodes().length>10);
    await page.getByRole('button',{name:'暂停漂动',exact:true}).click();
    const id = await page.locator('#node-search option').nth(1).getAttribute('value');
    await page.getByRole('combobox',{name:'定位图中实体'}).selectOption(id);
    await page.waitForTimeout(450);
    const split = await state();
    await page.getByRole('button',{name:'展开图谱',exact:true}).click({timeout:2500});
    await page.waitForTimeout(200);
    const expanded = await state();
    assert(!await page.locator('.answer-panel').isVisible(),'Expanded mode hides the answer');
    assert(Math.abs(expanded.panel.x-expanded.grid.x)<1 && Math.abs(expanded.panel.width-expanded.grid.width)<1,'No old answer column or gutter');
    assert(Math.abs(expanded.canvas.width-expanded.panel.width)<1,'Canvas fills the graph panel even with details open');
    assert(expanded.canvas.height>split.canvas.height,'Expanded canvas must be taller');
    assert(expanded.visible===expanded.nodes.length,'Entering expanded mode fits all loaded nodes');
    assert(expanded.selected[0]===split.selected[0],'Selection survives mode change');
    const detail=await page.locator('.node-detail').boundingBox();
    assert(detail.width>=320&&detail.width<=380,'Expanded details use a restrained fixed width');
    await page.getByRole('button',{name:'关闭实体详情'}).click();
    await page.waitForTimeout(100);
    const labels = await page.evaluate(()=>{const cy=document.querySelector('.graph-canvas')._cyreg.cy;return {shown:cy.nodes('.label-visible').length,total:cy.nodes().length};});
    assert(labels.shown>0&&labels.shown<labels.total,'Overview labels remain selective');
    await page.locator('.graph-canvas').scrollIntoViewIfNeeded();
    const target=await page.evaluate(id=>{const el=document.querySelector('.graph-canvas');const cy=el._cyreg.cy;const p=cy.getElementById('node:'+id).renderedPosition();const r=el.getBoundingClientRect();return {x:r.x+p.x,y:r.y+p.y};},id);
    await page.mouse.move(target.x,target.y);
    await page.mouse.click(target.x,target.y);
    await page.waitForTimeout(450);
    const focus=await page.evaluate(()=>{
      const cy=document.querySelector('.graph-canvas')._cyreg.cy;
      const n=cy.nodes(':selected').first(); const p=n.renderedPosition();
      return {x:p.x,y:p.y,width:cy.width(),height:cy.height(),zoom:cy.zoom(),color:n.style('border-color'),dim:cy.nodes('.dimmed').length,labels:cy.nodes().filter(n=>n.style('label')!=='').length};
    });
    assert(focus.zoom>expanded.zoom,'Selecting a node zooms in');
    assert(focus.x>40 && focus.x<focus.width-detail.width-12 && Math.abs(focus.y-focus.height/2)<30,'Focused node stays in the unobscured canvas');
    assert(focus.dim>0,'Unrelated nodes become subdued');
    assert(focus.color!=='rgb(24,46,37)','No old heavy dark selection border');
    assert(await page.getByRole('combobox',{name:'定位图中实体'}).inputValue()===id,'Canvas click selects the real node');
    await page.screenshot({path:'output/playwright/graph-expanded.png',fullPage:true});
    await page.getByRole('button',{name:'收起图谱',exact:true}).click();
    await page.waitForTimeout(200);
    const collapsed=await state();
    assert(await page.locator('.answer-panel').isVisible(),'Collapse restores answer');
    assert(Math.abs(collapsed.zoom-split.zoom)<0.001,'Collapse restores the original split zoom');
    assert(collapsed.selected[0]===split.selected[0],'Collapse retains selection');
    assert(JSON.stringify(collapsed.nodes)===JSON.stringify(split.nodes),'Toggle does not recreate or relayout nodes');
    await page.getByRole('button',{name:'恢复漂动',exact:true}).waitFor();
    await page.getByRole('button',{name:'展开图谱',exact:true}).click();
    await page.setViewportSize({width:390,height:844});
    await page.waitForTimeout(200);
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Mobile has no horizontal overflow');
    await page.screenshot({path:'output/playwright/graph-expanded-mobile.png',fullPage:true});
    return {passed:['full-width expansion','overlay details','auto fit','node focus','selection layers','collapse restores view and node positions','pause retained','mobile layout']};
  } finally {
    await page.setViewportSize({width:1440,height:1050});
    await page.reload();
  }
}
