async (page) => {
  const assert=(ok,message)=>{if(!ok)throw new Error(message)};
  const viewport=()=>page.evaluate(()=>{const cy=document.querySelector('.graph-canvas')._cyreg.cy;return {zoom:cy.zoom(),pan:cy.pan()}});
  const assertSameView=async(before)=>{const after=await viewport();assert(Math.abs(before.zoom-after.zoom)<0.0001&&Math.abs(before.pan.x-after.pan.x)<0.01&&Math.abs(before.pan.y-after.pan.y)<0.01,'Dismissing focus must preserve zoom and pan');};
  const clickSelected=async()=>{
    await page.locator('.graph-canvas').scrollIntoViewIfNeeded();
    const before=await viewport();
    const p=await page.evaluate(()=>{const el=document.querySelector('.graph-canvas');const cy=el._cyreg.cy;const n=cy.nodes(':selected').first();const p=n.renderedPosition();const r=el.getBoundingClientRect();return {x:r.x+p.x,y:r.y+p.y}});
    await page.mouse.click(p.x,p.y);
    await page.waitForTimeout(400);
    await assertSameView(before);
  };
  const assertOverview=async()=>{
    assert(!await page.locator('.node-detail').count(),'Second click must close the detail panel');
    const s=await page.evaluate(()=>{const cy=document.querySelector('.graph-canvas')._cyreg.cy;return {selected:cy.elements(':selected').length,dimmed:cy.elements('.dimmed').length,total:cy.nodes().length,visible:cy.nodes().filter(n=>{const p=n.renderedPosition();return p.x>=0&&p.y>=0&&p.x<=cy.width()&&p.y<=cy.height()}).length}});
    assert(s.selected===0&&s.dimmed===0,'Overview clears selection and dimming');
  };
  try {
    await page.setViewportSize({width:1440,height:1050});
    await page.reload();
    await page.waitForFunction(()=>document.querySelector('.graph-canvas')?._cyreg?.cy?.nodes().length>10);
    await page.getByRole('button',{name:'暂停漂动',exact:true}).click();
    await page.getByRole('button',{name:'展开图谱',exact:true}).click();
    const id=await page.locator('#node-search option').nth(1).getAttribute('value');
    await page.getByRole('combobox',{name:'定位图中实体'}).selectOption(id);
    await page.waitForTimeout(400);
    await clickSelected();
    await assertOverview();
    await page.getByRole('combobox',{name:'定位图中实体'}).selectOption(id);
    await page.waitForTimeout(400);
    const panel=page.getByRole('complementary',{name:'节点详情'});
    await panel.getByText('关系数',{exact:true}).waitFor();
    await panel.getByText('节点描述',{exact:true}).waitFor();
    await panel.getByText(/相关实体（\d+）/).waitFor();
    assert(await panel.locator('.entity-avatar').count()===1,'Detail has a type-colored identity mark');
    const neighbor=panel.locator('.node-neighbors button').first();
    const nextId=await neighbor.getAttribute('data-node-id');
    await neighbor.click();
    await page.waitForTimeout(400);
    assert(await page.locator('#node-search').inputValue()===nextId,'Related entity links focus the other node');
    const beforeClose=await viewport();
    await panel.getByRole('button',{name:'关闭实体详情'}).click();
    await assertSameView(beforeClose);
    await assertOverview();
    await page.getByRole('combobox',{name:'定位图中实体'}).selectOption(id);
    await page.waitForTimeout(400);
    await page.getByRole('button',{name:'适应画布',exact:true}).click();
    await assertOverview();
    await page.getByRole('combobox',{name:'定位图中实体'}).selectOption(id);
    await page.waitForTimeout(400);
    await page.screenshot({path:'output/playwright/node-detail-refined.png',fullPage:true});
    await page.getByRole('button',{name:'收起图谱',exact:true}).click();
    await page.waitForTimeout(200);
    await page.getByRole('button',{name:'适应画布',exact:true}).click();
    await page.getByRole('combobox',{name:'定位图中实体'}).selectOption(id);
    await page.waitForTimeout(400);
    await clickSelected();
    await assertOverview();
    return {passed:['repeat click clears emphasis without changing viewport in both modes','close preserves viewport','fit clears focus','structured detail','related entity navigation']};
  } finally { await page.reload(); }
}
