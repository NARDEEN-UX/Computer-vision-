import { useMemo, useState } from 'react'
import ImageUploader from './components/ImageUploader'
import ParameterPanel from './components/ParameterPanel'
import ResultCard from './components/ResultCard'
import HistogramCharts from './components/HistogramCharts'
import { dataUrlToFile, processImage } from './services/api'

const OPERATION_GROUPS = [
  {
    key: 'noise', label: 'Noise', note: 'Image degradation models', operations: [
      { group: 'noise', name: 'gaussian', label: 'Gaussian Noise', description: 'Adds normally distributed intensity variation to each pixel.' },
      { group: 'noise', name: 'salt-pepper', label: 'Salt & Pepper Noise', description: 'Replaces selected pixels with black or white impulse noise.' },
    ],
  },
  {
    key: 'filters', label: 'Spatial Filtering', note: 'Neighborhood-based smoothing', operations: [
      { group: 'filters', name: 'average', label: 'Average Filter', description: 'Smooths the image by averaging the values in a local neighborhood.' },
      { group: 'filters', name: 'gaussian', label: 'Gaussian Filter', description: 'Smooths the image using a Gaussian-weighted neighborhood.' },
      { group: 'filters', name: 'median', label: 'Median Filter', description: 'Replaces each pixel with the median value of its local neighborhood.' },
    ],
  },
  {
    key: 'edge', label: 'Edge Detection', note: 'Gradient and contour operators', operations: [
      { group: 'edge', name: 'sobel', label: 'Sobel', description: 'Estimates horizontal and vertical intensity change using weighted 3×3 kernels.' },
      { group: 'edge', name: 'roberts', label: 'Roberts', description: 'Estimates diagonal intensity change using compact 2×2 kernels.' },
      { group: 'edge', name: 'prewitt', label: 'Prewitt', description: 'Estimates horizontal and vertical intensity change using uniform 3×3 kernels.' },
      { group: 'edge', name: 'canny', label: 'Canny', description: 'Detects contours with the OpenCV Canny implementation and two thresholds.' },
    ],
  },
  {
    key: 'histogram', label: 'Histogram', note: 'Intensity analysis', operations: [
      { group: 'histogram', name: 'analyze', label: 'Histogram & Distribution', description: 'Counts intensity values and calculates their normalized probability distribution.' },
    ],
  },
  {
    key: 'transform',
    label: 'Transform & Enhancement',
    note: 'Color and contrast operations',
    operations: [
      {
        group: 'transform',
        name: 'equalize',
        label: 'Histogram Equalization',
        description: 'Redistributes intensity values using the cumulative distribution function.',
      },
      {
        group: 'transform',
        name: 'to-gray',
        label: 'RGB to Grayscale',
        description: 'Converts an RGB image to one grayscale channel.',
      },
      {
        group: 'transform',
        name: 'to-rgb',
        label: 'Grayscale to RGB',
        description: 'Converts a grayscale image into three RGB channels.',
      },
      {
        group: 'transform',
        name: 'normalize',
        label: 'Image Normalization',
        description: 'Maps the selected input intensity range to the full 0–255 display range.',
      },
    ],
  },
]

const OPERATIONS = OPERATION_GROUPS.flatMap((group) => group.operations)
const DEFAULTS = {
  'noise/gaussian': { mean: 0, sigma: 20, seed: '' },
  'noise/salt-pepper': { density: 0.05, salt_probability: 0.5, seed: '' },
  'filters/average': { kernel_size: 3 },
  'filters/gaussian': { kernel_size: 3, sigma: 1 },
  'filters/median': { kernel_size: 3 },
  'edge/sobel': { direction: 'magnitude' },
  'edge/roberts': { direction: 'magnitude' },
  'edge/prewitt': { direction: 'magnitude' },
  'edge/canny': { lower_threshold: 50, upper_threshold: 150 },
  'histogram/analyze': {},
  'transform/equalize': {},
  'transform/to-gray': {},
  'transform/to-rgb': {},
  'transform/normalize': { min_val: 0, max_val: 255 },
}

function operationId(operation) { return `${operation.group}/${operation.name}` }
function number(value) { return value === '' ? value : Number(value) }

function validate(operation, values) {
  if (operation.group === 'noise' && operation.name === 'gaussian' && Number(values.sigma) <= 0) return 'Sigma must be greater than zero.'
  if (operation.name === 'salt-pepper' && (Number(values.density) < 0 || Number(values.density) > 1)) return 'Density must be between 0 and 1.'
  if (operation.name === 'salt-pepper' && (Number(values.salt_probability) < 0 || Number(values.salt_probability) > 1)) return 'Salt probability must be between 0 and 1.'
  if (operation.group === 'noise' && values.seed !== '' && !Number.isInteger(Number(values.seed))) return 'Seed must be an integer.'
  if (operation.group === 'filters' && operation.name === 'gaussian' && Number(values.sigma) <= 0) return 'Sigma must be greater than zero.'
  if (operation.group === 'edge' && operation.name === 'canny' && (Number(values.lower_threshold) < 0 || Number(values.lower_threshold) >= Number(values.upper_threshold))) return 'Lower threshold must be less than upper threshold.'
  if (operation.group === 'transform' && operation.name === 'normalize' && Number(values.max_val) <= Number(values.min_val)) return 'Maximum value must be greater than minimum value.'
  return ''
}

function HistogramSourceImages({ originalUrl, originalName, workingUrl, workingLabel, selectedSourceLabel }) {
  return <div className="histogram-source-images">
    <article className="image-card">
      <div className="result-heading"><div><span className="eyebrow">Original</span><h3>{originalName || 'Uploaded image'}</h3></div></div>
      {originalUrl ? <img src={originalUrl} alt="Original uploaded image" /> : <div className="empty-image">Upload an image to begin</div>}
      <div className="image-caption">Original uploaded image</div>
    </article>
    {workingUrl && <article className={`image-card ${selectedSourceLabel === workingLabel ? 'selected-source-card' : ''}`}>
      <div className="result-heading"><div><span className="eyebrow">Latest result</span><h3>{workingLabel}</h3></div></div>
      <img src={workingUrl} alt={`${workingLabel} result`} />
      <div className="image-caption">{selectedSourceLabel === workingLabel ? 'Currently selected for analysis' : 'Available as a later input'}</div>
    </article>}
  </div>
}

export default function App() {
  const [file, setFile] = useState(null)
  const [originalUrl, setOriginalUrl] = useState('')
  const [workingFile, setWorkingFile] = useState(null)
  const [workingUrl, setWorkingUrl] = useState('')
  const [workingLabel, setWorkingLabel] = useState('Original image')
  const [sourceMode, setSourceMode] = useState('original')
  const [selectedCategory, setSelectedCategory] = useState('noise')
  const [operationKey, setOperationKey] = useState('noise/gaussian')
  const [values, setValues] = useState(DEFAULTS['noise/gaussian'])
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const operation = useMemo(() => OPERATIONS.find((item) => operationId(item) === operationKey), [operationKey])
  const activeGroup = useMemo(() => OPERATION_GROUPS.find((group) => group.key === selectedCategory) ?? OPERATION_GROUPS[0], [selectedCategory])
  const selectedSource = sourceMode === 'latest' && workingFile ? workingFile : file
  const selectedSourceLabel = sourceMode === 'latest' && workingFile ? workingLabel : 'Original image'

  function chooseFile(nextFile) {
    setError('')
    if (!nextFile) return
    if (!nextFile.type.startsWith('image/')) return setError('Please choose a PNG or JPEG image.')
    setFile(nextFile); setOriginalUrl(URL.createObjectURL(nextFile)); setWorkingFile(null); setWorkingUrl(''); setWorkingLabel('Original image'); setSourceMode('original'); setResult(null)
  }

  function chooseOperation(nextOperation) {
    const nextKey = operationId(nextOperation)
    setSelectedCategory(nextOperation.group); setOperationKey(nextKey); setValues({ ...DEFAULTS[nextKey] }); setError('')
  }

  function chooseCategory(categoryKey) {
    const group = OPERATION_GROUPS.find((item) => item.key === categoryKey)
    if (group?.operations[0]) chooseOperation(group.operations[0])
  }

  function updateValue(key, value) {
    const numericParameter = key !== 'direction'
    setValues((current) => ({ ...current, [key]: numericParameter ? number(value) : value }))
  }

  async function apply() {
    setError('')
    if (!selectedSource) return setError('Upload an image first.')
    const validationError = validate(operation, values)
    if (validationError) return setError(validationError)
    setLoading(true)
    try {
      const response = await processImage(operation, selectedSource, values)
      if (operation.group === 'histogram') {
        setResult({ ...response, label: operation.label, sourceLabel: selectedSourceLabel, kind: 'histogram' })
        return
      }
      setWorkingFile(dataUrlToFile(response.image, `${operation.name}-result.png`)); setWorkingUrl(response.image); setWorkingLabel(operation.label); setSourceMode('latest')
      setResult({ ...response, label: operation.label, sourceLabel: selectedSourceLabel })
    } catch (requestError) {
      const detail = requestError.response?.data?.detail
      setError(typeof detail === 'string' ? detail : 'The image could not be processed. Check that the backend is running.')
    } finally { setLoading(false) }
  }

  return <main className="page-shell">
    <header className="hero"><div className="hero-copy"><span className="eyebrow"></span><h1>Image Processing</h1><p className="hero-subtitle">Computer Vision </p><p className="hero-description">Interactive experimentation with custom image-processing algorithms.</p></div><div className="scope-chip"><span>Laboratory scope</span><b>Noise · filtering · edges</b></div></header>
    <section className="lab-layout">
      <aside className="control-panel">
        <div className="step-heading"><span className="step-number">01</span><div><span className="section-label">Input</span><h2>Image source</h2></div></div>
        <ImageUploader file={file} onChange={chooseFile} />
        <div className="step-heading"><span className="step-number">02</span><div><span className="section-label">Operation</span><h2>Select a method</h2></div></div>
        <div className="operation-picker">
          <label className="category-field" htmlFor="operation-category"><span>Category</span><select id="operation-category" value={selectedCategory} onChange={(event) => chooseCategory(event.target.value)}>{OPERATION_GROUPS.map((group) => <option value={group.key} key={group.key}>{group.label}</option>)}</select></label>
          <div className="operation-list">{activeGroup.operations.map((item) => <button className={operationId(item) === operationKey ? 'operation-option selected' : 'operation-option'} key={operationId(item)} onClick={() => chooseOperation(item)} type="button"><span className="operation-mark" /><span>{item.label}</span><span className="operation-arrow">→</span></button>)}</div>
        </div>
        <div className="step-heading parameter-step"><span className="step-number">03</span><div><span className="section-label">Parameters</span><h2>{operation.label}</h2></div></div>
        <p className="operation-description">{operation.description}</p>
        <ParameterPanel operation={operation} values={values} onChange={updateValue} />
        {workingFile && <div className="source-control"><span className="field-caption">Process this operation on</span><div className="source-options"><button type="button" className={sourceMode === 'original' ? 'source-option active' : 'source-option'} onClick={() => setSourceMode('original')}>Original</button><button type="button" className={sourceMode === 'latest' ? 'source-option active' : 'source-option'} onClick={() => setSourceMode('latest')}>Latest result</button></div><small>{selectedSourceLabel}</small></div>}
        {error && <div className="error-message" role="alert">{error}</div>}
        <button className="apply-button" disabled={loading} onClick={apply}>{loading ? 'Processing…' : 'Run operation'}</button>
      </aside>
      <section className="results-area"><div className="results-header"><div className="step-heading"><span className="step-number">04</span><div><span className="section-label">Result</span><h2>Visual comparison</h2></div></div>{result && <span className="result-status">Ready · {result.processing_time_ms} ms</span>}</div>{result?.kind === 'histogram' ? <div className="histogram-analysis-layout"><HistogramSourceImages originalUrl={originalUrl} originalName={file?.name} workingUrl={workingUrl} workingLabel={workingLabel} selectedSourceLabel={result.sourceLabel} /><HistogramCharts result={result} /></div> : <div className="image-grid"><article className="image-card"><div className="result-heading"><div><span className="eyebrow">Reference</span><h3>{file ? file.name : 'No image selected'}</h3></div></div>{originalUrl ? <img src={originalUrl} alt="Original uploaded image" /> : <div className="empty-image">Upload an image to begin</div>}<div className="image-caption">Original uploaded image</div></article>{result ? <ResultCard result={result} /> : <div className="empty-result"><span>04</span><h2>Run a method to inspect the result</h2><p>The processed image and technical metadata will appear here.</p></div>}</div>}{result?.histograms && <div className="comparison-charts"><HistogramCharts result={result.histograms.before} title={`Before ${result.label.toLowerCase()}`} /><HistogramCharts result={result.histograms.after} title={`After ${result.label.toLowerCase()}`} /></div>}{workingUrl && result && result.kind !== 'histogram' && <div className="working-strip"><span className="eyebrow">Current working image</span><strong>{workingLabel}</strong><span>Next filters can use this result as their input.</span></div>}</section>
    </section>
  </main>
}
