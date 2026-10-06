const KERNELS = [3, 5, 7, 9]

function NumberControl({ label, value, onChange, min, max, step = 1, help }) {
  return <label className="field"><span>{label}</span><div className="control-row"><input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(event.target.value)} /><input className="number-input" type="number" min={min} max={max} step={step} value={value} onChange={(event) => onChange(event.target.value)} /></div>{help && <small>{help}</small>}</label>
}

function SeedControl({ value, onChange }) { return <label className="field"><span>Random seed <em>optional</em></span><input type="number" step="1" value={value} placeholder="Leave empty for random" onChange={(event) => onChange(event.target.value)} /></label> }
function KernelControl({ value, onChange }) { return <label className="field"><span>Kernel size</span><select value={value} onChange={(event) => onChange(Number(event.target.value))}>{KERNELS.map((size) => <option key={size} value={size}>{size} × {size}</option>)}</select></label> }
function DirectionControl({ value, onChange }) { return <label className="field"><span>Direction</span><select value={value} onChange={(event) => onChange(event.target.value)}><option value="magnitude">Magnitude</option><option value="x">X direction</option><option value="y">Y direction</option></select></label> }

export default function ParameterPanel({ operation, values, onChange }) {
  if (operation.group === 'noise' && operation.name === 'gaussian') return <div className="fields"><NumberControl label="Mean (μ)" value={values.mean} min={-255} max={255} step={1} onChange={(value) => onChange('mean', value)} /><NumberControl label="Sigma (σ)" value={values.sigma} min={0.1} max={100} step={0.1} onChange={(value) => onChange('sigma', value)} help="Controls the strength of the added noise." /><SeedControl value={values.seed} onChange={(value) => onChange('seed', value)} /></div>
  if (operation.group === 'noise' && operation.name === 'salt-pepper') return <div className="fields"><NumberControl label="Density" value={values.density} min={0} max={1} step={0.01} onChange={(value) => onChange('density', value)} help="Fraction of pixels replaced." /><NumberControl label="Salt probability" value={values.salt_probability} min={0} max={1} step={0.01} onChange={(value) => onChange('salt_probability', value)} /><SeedControl value={values.seed} onChange={(value) => onChange('seed', value)} /></div>
  if (operation.name === 'gaussian') return <div className="fields"><KernelControl value={values.kernel_size} onChange={(value) => onChange('kernel_size', value)} /><NumberControl label="Sigma (σ)" value={values.sigma} min={0.1} max={10} step={0.1} onChange={(value) => onChange('sigma', value)} help="Controls the spread of the Gaussian weights." /></div>
  if (operation.group === 'edge' && operation.name === 'canny') return <div className="fields"><NumberControl label="Lower threshold" value={values.lower_threshold} min={0} max={255} step={1} onChange={(value) => onChange('lower_threshold', value)} /><NumberControl label="Upper threshold" value={values.upper_threshold} min={1} max={255} step={1} onChange={(value) => onChange('upper_threshold', value)} help="The upper threshold must be greater than the lower threshold." /></div>
  if (operation.group === 'edge') return <div className="fields"><DirectionControl value={values.direction} onChange={(value) => onChange('direction', value)} /></div>
  if (operation.group === 'histogram') return <div className="analysis-note">Histogram and distribution are calculated automatically from the uploaded image.</div>
  return <div className="fields"><KernelControl value={values.kernel_size} onChange={(value) => onChange('kernel_size', value)} /></div>
}
