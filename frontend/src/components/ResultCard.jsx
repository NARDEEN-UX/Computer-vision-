const PARAMETER_LABELS = { kernel_size: 'Kernel', sigma: 'Sigma', mean: 'Mean', density: 'Density', salt_probability: 'Salt probability', seed: 'Seed' }

export default function ResultCard({ result }) {
  const parameters = Object.entries(result.parameters).filter(([, value]) => value !== null && value !== undefined && value !== '')
  return <article className="result-card"><div className="result-heading"><div><span className="eyebrow">Processed result</span><h3>{result.label}</h3></div><span className="time-badge">{result.processing_time_ms} ms</span></div><img src={result.image} alt={`${result.label} result`} /><div className="image-caption">Input: {result.sourceLabel}</div><div className="parameter-summary">{parameters.map(([key, value]) => <span key={key}>{PARAMETER_LABELS[key] ?? key}: <b>{key === 'kernel_size' ? `${value} × ${value}` : String(value)}</b></span>)}</div></article>
}
