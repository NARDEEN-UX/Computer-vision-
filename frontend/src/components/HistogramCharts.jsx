import { useState } from 'react'

const CHANNELS = {
  red: { label: 'Red', color: '#cc5b54' },
  green: { label: 'Green', color: '#57936b' },
  blue: { label: 'Blue', color: '#537fb4' },
  grayscale: { label: 'Grayscale', color: '#48575b' },
}

function Legend({ names }) {
  return <div className="chart-legend">{names.map((name) => <span key={name}><i style={{ background: CHANNELS[name].color }} />{CHANNELS[name].label}</span>)}</div>
}

function formatValue(value) {
  if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`
  if (value >= 1000) return `${(value / 1000).toFixed(1)}k`
  if (value < 1) return value.toFixed(2)
  return Math.round(value).toString()
}

function linearTicks(maximum) {
  return [0, maximum / 2, maximum]
}

function logarithmicTicks(maximum) {
  const ticks = [0]
  for (let value = 1; value < maximum; value *= 10) ticks.push(value)
  if (ticks[ticks.length - 1] !== maximum) ticks.push(maximum)
  return ticks
}

function Plot({ title, yLabel, channels, mode, scale, onScaleChange }) {
  const width = 760
  const height = 300
  const left = 72
  const right = 18
  const top = 18
  const bottom = 42
  const plotWidth = width - left - right
  const plotHeight = height - top - bottom
  const names = Object.keys(channels)
  const values = names.flatMap((name) => channels[name][mode])
  // The backend returns probabilities in [0, 1]. Distribution charts display
  // those values as percentages, while the backend data stays unchanged.
  const isProbability = mode === 'distribution' || mode === 'cdf'
  const displayValues = isProbability ? values.map((value) => value * 100) : values
  const dataMaximum = Math.max(...displayValues, isProbability ? 0.01 : 1)
  const chartMaximum = isProbability ? dataMaximum * 1.1 : dataMaximum
  const transform = (value) => scale === 'log' ? Math.log10(value + 1) : value
  const transformedMaximum = transform(chartMaximum)
  const x = (index) => left + (index / 255) * plotWidth
  const y = (value) => top + plotHeight - (transform(value) / transformedMaximum) * plotHeight
  const linePath = (series) => series.map((value, index) => {
    const displayValue = isProbability ? value * 100 : value
    return `${index === 0 ? 'M' : 'L'} ${x(index).toFixed(2)} ${y(displayValue).toFixed(2)}`
  }).join(' ')
  const xTicks = [0, 64, 128, 192, 255]
  const yTicks = scale === 'log' ? logarithmicTicks(dataMaximum) : linearTicks(dataMaximum)
  const barWidth = Math.max((plotWidth / 256) * 1.6, 2)

  return <article className="histogram-card">
    <div className="chart-heading">
      <div><span className="eyebrow">{mode === 'histogram' ? 'Histogram' : mode === 'cdf' ? 'CDF' : 'PDF'}</span><h3>{title}</h3></div>
      <div className="chart-tools">
        {mode === 'histogram' && <div className="scale-toggle" aria-label="Histogram vertical scale">
          <button className={scale === 'linear' ? 'active' : ''} onClick={() => onScaleChange('linear')}>Linear</button>
          <button className={scale === 'log' ? 'active' : ''} onClick={() => onScaleChange('log')}>Log</button>
        </div>}
        <Legend names={names} />
      </div>
    </div>
    <svg className="histogram-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${title}, ${yLabel}`}>
      {yTicks.map((tick) => <g key={`y-${tick}`}>
        <line x1={left} y1={y(tick)} x2={left + plotWidth} y2={y(tick)} className="chart-grid" />
        <text x={left - 8} y={y(tick) + 3} textAnchor="end" className="chart-tick">{formatValue(tick)}</text>
      </g>)}
      <line x1={left} y1={top} x2={left} y2={top + plotHeight} className="chart-axis" />
      <line x1={left} y1={top + plotHeight} x2={left + plotWidth} y2={top + plotHeight} className="chart-axis" />
      {xTicks.map((tick) => <g key={tick}><line x1={x(tick)} y1={top + plotHeight} x2={x(tick)} y2={top + plotHeight + 5} className="chart-axis" /><text x={x(tick)} y={height - 14} textAnchor="middle" className="chart-tick">{tick}</text></g>)}
      <text x={left + plotWidth / 2} y={height - 1} textAnchor="middle" className="chart-axis-label">Intensity</text>
      <text x="14" y={top + plotHeight / 2} textAnchor="middle" transform={`rotate(-90 14 ${top + plotHeight / 2})`} className="chart-axis-label">{yLabel}</text>
      {mode === 'histogram'
        ? names.map((name) => <g key={name}>{channels[name][mode].map((value, index) => <rect key={index} x={x(index) - barWidth / 2} y={y(value)} width={barWidth} height={top + plotHeight - y(value)} fill={CHANNELS[name].color} stroke={CHANNELS[name].color} strokeWidth="0.6" opacity={names.length > 1 ? 0.48 : 0.82} />)}</g>)
        : names.map((name) => <path key={name} d={linePath(channels[name][mode])} fill="none" stroke={CHANNELS[name].color} strokeWidth="2.2" />)}
    </svg>
  </article>
}

export default function HistogramCharts({ result, title }) {
  const isRgb = result.image_type === 'rgb'
  const names = isRgb ? ['red', 'green', 'blue'] : ['grayscale']
  const channels = Object.fromEntries(names.map((name) => [name, result.channels[name]]))
  const titlePrefix = isRgb ? 'RGB' : 'Grayscale'
  const [histogramScale, setHistogramScale] = useState('linear')

  return <div className="histogram-results">
    <div className="analysis-summary"><span className="eyebrow">Image analysis</span><strong>{title ?? `${titlePrefix} image`}{result.metadata ? ` · ${result.metadata.width} × ${result.metadata.height}` : ''}</strong><span>Computed with a custom 256-bin histogram, PDF, and CDF.</span></div>
    <Plot title={`${titlePrefix} Histogram`} yLabel="Pixel Count" channels={channels} mode="histogram" scale={histogramScale} onScaleChange={setHistogramScale} />
    <Plot title={`${titlePrefix} PDF`} yLabel="Probability (%)" channels={channels} mode="distribution" scale="linear" onScaleChange={() => {}} />
    <Plot title={`${titlePrefix} CDF`} yLabel="Cumulative Probability (%)" channels={channels} mode="cdf" scale="linear" onScaleChange={() => {}} />
  </div>
}
