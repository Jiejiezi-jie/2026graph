async (page) => {
  const assert=(ok,message)=>{if(!ok)throw new Error(message)};
  let submitted=[];
  await page.route('**/api/query',route=>{submitted.push(route.request().postDataJSON());return route.fulfill({status:400,json:{error:{message:'离线控件测试'}}});});
  try {
    await page.setViewportSize({width:1440,height:1050});
    await page.reload();
    const mode=page.getByRole('combobox',{name:'检索模式',exact:true});
    await mode.selectOption('mix');
    const input=page.getByRole('spinbutton',{name:'Top-K',exact:true});
    await page.getByRole('button',{name:'增加 Top-K',exact:true}).click({timeout:2500});
    assert(await input.inputValue()==='6','Increment updates the value');
    await page.getByRole('button',{name:'减少 Top-K',exact:true}).click();
    assert(await input.inputValue()==='5','Decrement updates the value');
    await input.fill('1');
    assert(await page.getByRole('button',{name:'减少 Top-K',exact:true}).isDisabled(),'Lower bound is enforced');
    await input.fill('50');
    assert(await page.getByRole('button',{name:'增加 Top-K',exact:true}).isDisabled(),'Upper bound is enforced');
    const hints=[];
    for(const value of ['local','global','hybrid','mix','naive']) {await mode.selectOption(value);hints.push(await page.locator('.query-hint').innerText());}
    assert(new Set(hints).size===5,'Every mode has its own explanation');
    await input.fill('');
    await input.blur();
    assert(await input.inputValue()==='5','Blank input gets a usable default on blur');
    await input.fill('7');
    await page.getByRole('textbox',{name:'提出你的问题'}).fill('离线控件测试');
    await page.getByRole('button',{name:'开始查询'}).click();
    await page.getByRole('alert').filter({hasText:'离线控件测试'}).waitFor();
    assert(submitted.length===1&&submitted[0].top_k===7&&submitted[0].options.mode==='naive','Controls preserve the query payload contract');
    await page.getByRole('button',{name:'关闭提示'}).click();
    await mode.selectOption('mix');
    await page.locator('.query-composer').screenshot({path:'output/playwright/retrieval-controls.png'});
    await page.setViewportSize({width:390,height:844});
    await page.waitForTimeout(200);
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Controls fit narrow screens');
    await page.locator('.query-composer').screenshot({path:'output/playwright/retrieval-controls-mobile.png'});
    return {passed:['stepper and limits','editable numeric input','five distinct mode explanations','query payload','mobile layout'],paidQueries:0};
  } finally {await page.unroute('**/api/query');await page.setViewportSize({width:1440,height:1050});await page.reload();}
}
