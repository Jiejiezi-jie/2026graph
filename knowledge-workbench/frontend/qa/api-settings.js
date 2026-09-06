async (page) => {
  // Use the separate ephemeral QA backend, never overwrite the user's connection.
  const assert = (ok, message) => { if (!ok) throw new Error(message) }
  const key = 'sk-browser-qa-only';
  await page.route('**/api/**', async route => {
    const path = '/api/' + route.request().url().split('/api/')[1];
    const response = await route.fetch({ url: 'http://127.0.0.1:8001' + path });
    await route.fulfill({ response });
  });
  try {
    await page.reload();
    await page.getByRole('button', { name: 'API 设置', exact: true }).click();
    const dialog = page.getByRole('dialog', { name: 'API 设置', exact: true });
    const urlInput = dialog.getByLabel('Base URL', { exact: true });
    const keyInput = dialog.getByLabel('API Key', { exact: true });
    await page.waitForFunction(() => !document.querySelector('#api-key').disabled);
    assert(await urlInput.inputValue() === 'https://api.deepseek.com', 'Default must be DeepSeek');
    assert(await keyInput.getAttribute('type') === 'password', 'Key must be masked');
    await keyInput.fill(key);
    await dialog.getByRole('button', { name: '保存并应用' }).click();
    await dialog.getByRole('status').filter({hasText: '已应用于本次后端运行'}).waitFor();
    assert(await keyInput.inputValue() === '', 'Saved key must be cleared from input');
    await dialog.getByRole('button', { name: '关闭 API 设置', exact: true }).click();
    await page.getByRole('button', { name: 'API 设置', exact: true }).click();
    await dialog.getByText('● Key 已配置', { exact: true }).waitFor();
    assert(await keyInput.inputValue() === '', 'Reopening must not retrieve the key');
    await urlInput.fill('https://api.deepseek.com/?credential=bad');
    await dialog.getByRole('button', { name: '保存并应用' }).click();
    await dialog.getByRole('alert').waitFor();
    await dialog.getByRole('button', { name: '使用 DeepSeek 默认地址' }).click();
    await dialog.getByRole('button', { name: '保存并应用' }).click();
    await dialog.getByRole('status').filter({hasText: '已应用于本次后端运行'}).waitFor();
    await page.setViewportSize({ width: 1440, height: 1050 });
    await dialog.screenshot({path: 'output/playwright/api-settings-desktop.png'});
    await page.setViewportSize({ width: 375, height: 812 });
    await dialog.screenshot({path: 'output/playwright/api-settings-mobile.png'});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'No mobile overflow');
    assert(await page.evaluate(key => !JSON.stringify(localStorage).includes(key) && !JSON.stringify(sessionStorage).includes(key), key), 'Do not persist keys in browser storage');
    await page.keyboard.press('Escape');
    assert(await page.getByRole('dialog', { name: 'API 设置', exact: true }).count() === 0, 'Escape closes dialog');
    return { passed: ['DeepSeek default', 'masked key', 'real backend save', 'clear saved input', 'reopen without secret', 'invalid URL', 'keep key on blank input', 'mobile layout', 'no browser storage', 'Escape closes dialog'] };
  } finally {
    await page.unroute('**/api/**');
    await page.setViewportSize({width: 1440, height: 1050});
    await page.reload();
  }
}
