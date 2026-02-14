import React, { useState } from 'react'
import './App.css'

export default function App() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const API_BASE = 'https://email-spam-classifier-react.onrender.com'

async function handleSubmit(e) {
  e.preventDefault()
  setError(null)
  setResult(null)

  if (!file) {
    setError('Please choose a PDF first.')
    return
  }

  const form = new FormData()
  form.append('file', file)

  setLoading(true)
  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      body: form
    })

    const data = await res.json()
    if (!res.ok) throw new Error(data.error || 'Prediction failed')
    setResult(data)

  } catch (err) {
    setError(err.message)
  } finally {
    setLoading(false)
  }
}

  return (
    <div className="container">
      <h1>PDF Email Spam Classifier</h1>
      <p className="hint">Upload a PDF of an email. The model predicts whether it's SPAM or Non‑SPAM.</p>

      <form onSubmit={handleSubmit} className="card">
        <div className="row">
          <input
            type="file"
            accept="application/pdf"
            onChange={e => setFile(e.target.files?.[0] || null)}
          />
          <button className="button" disabled={loading} type="submit">
            {loading ? 'Analyzing…' : 'Classify'}
          </button>
        </div>
      </form>

      {error && (
        <div className="card" style={{ marginTop: 12 }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div className="card" style={{ marginTop: 12 }}>
          <div>
            <span className={`badge ${result.label === 'SPAM' ? 'spam' : 'ham'}`}>
              {result.label}
            </span>
            {typeof result.prob === 'number' && (
              <span style={{ marginLeft: 10 }}>
                Confidence: {(result.prob * 100).toFixed(1)}%
              </span>
            )}
          </div>
        </div>
      )}

      <div className="footer">
        <div>Backend: Flask • Frontend: React + Vite • Model: TF‑IDF + Logistic Regression</div>
      </div>
    </div>
  )
}
