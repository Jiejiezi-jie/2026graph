import { useEffect, useRef, useState, type FormEvent } from 'react'
import { LoaderCircle, X } from 'lucide-react'
import { api } from './api'

type ConnectionSettings = { base_url: string; api_key_configured: boolean; llm_model: string; session_override: boolean }
const deepseekUrl = 'https://api.deepseek.com'

export default function ApiSettings({ onClose, onSaved }: { onClose: () => void; onSaved: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null)
  const [current, setCurrent] = useState<ConnectionSettings | null>(null)
  const [baseUrl, setBaseUrl] = useState(deepseekUrl)
  const [key, setKey] = useState('')
  const [error, setError] = useState('')
  const [saved, setSaved] = useState(false)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    let active = true
    dialog.current?.showModal()
    api<ConnectionSettings>('/settings/connection').then(data => {
      if (active) { setCurrent(data); setBaseUrl(data.base_url) }
    }).catch(e => { if (active) setError(e.message) })
    return () => { active = false }
  }, [])

  async function save(event: FormEvent) {
    event.preventDefault()
    if (saving || !current) return
    setSaving(true); setError(''); setSaved(false)
    try {
      const data = await api<ConnectionSettings>('/settings/connection', {
        base_url: baseUrl.trim(), ...(key.trim() ? { api_key: key.trim() } : {}),
      })
      setCurrent(data); setBaseUrl(data.base_url); setKey(''); setSaved(true)
      onSaved()
    } catch (e) { setError((e as Error).message) }
    finally { setSaving(false) }
  }

  return <dialog ref={dialog} className="api-settings-dialog" aria-labelledby="api-settings-title"
    onCancel={event => { if (saving) event.preventDefault(); else onClose() }}>
    <div className="modal-heading"><h2 id="api-settings-title">API 设置</h2><button disabled={saving} aria-label="关闭 API 设置" onClick={onClose}><X size={20} /></button></div>
    <p className="muted">填写本次使用的模型接口。网页设置只在本次后端运行期间生效，重启后恢复启动配置。</p>
    {!current && !error && <p className="muted" role="status">正在读取配置…</p>}
    {error && <div className="alert error" role="alert">{error}</div>}
    <form onSubmit={save} autoComplete="off">
      <label className="field-label" htmlFor="api-base-url">Base URL</label>
      <input id="api-base-url" type="url" required maxLength={2048} value={baseUrl} disabled={saving || !current}
        onChange={e => { setBaseUrl(e.target.value); setSaved(false) }} spellCheck={false} autoCapitalize="none" aria-describedby="api-url-hint" />
      <div className="api-url-help"><span id="api-url-hint">兼容 OpenAI 的接口根地址</span><button type="button" disabled={saving || !current} onClick={() => { setBaseUrl(deepseekUrl); setSaved(false) }}>使用 DeepSeek 默认地址</button></div>
      <label className="field-label" htmlFor="api-key">API Key</label>
      <input id="api-key" type="password" autoComplete="off" spellCheck={false} autoCapitalize="none" maxLength={4096}
        required={!!current && !current.api_key_configured} value={key} disabled={saving || !current}
        placeholder={current?.api_key_configured ? '已配置；留空保留，输入新 Key 可替换' : '输入你的 API Key'}
        aria-describedby="api-key-hint" onChange={e => { setKey(e.target.value); setSaved(false) }} />
      <p id="api-key-hint" className="api-field-help">Key 仅保存在后端内存中，保存后不回显。</p>
      <div className="api-connection-summary"><span className={current?.api_key_configured ? 'success-text' : 'muted'}>{current?.api_key_configured ? '● Key 已配置' : '○ 尚未配置 Key'}</span><span>当前模型：{current?.llm_model || '—'}</span></div>
      {current && baseUrl.trim().replace(/\/+$/, '') !== current.base_url && <p className="warning-note">修改 URL 可能使已有索引的配置不匹配；保存后会刷新索引状态。单独更换 Key 不影响索引复用。</p>}
      {saved && <p className="api-save-success" role="status">已应用于本次后端运行，后续请求立即生效。尚未验证接口连通性。</p>}
      <div className="modal-actions"><button type="button" className="secondary" disabled={saving} onClick={onClose}>关闭</button><button className="primary" type="submit" disabled={saving || !current}>{saving && <LoaderCircle size={15} className="spin" />}{saving ? '保存中…' : '保存并应用'}</button></div>
    </form>
  </dialog>
}
