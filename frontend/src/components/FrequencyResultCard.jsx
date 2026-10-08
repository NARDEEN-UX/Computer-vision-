function FrequencyView({ title, image, description }) {
  return <article className="frequency-view">
    <div className="result-heading"><div><span className="eyebrow">Frequency domain</span><h3>{title}</h3></div></div>
    <img src={image} alt={`${title} Fourier visualization`} />
    <div className="image-caption">{description}</div>
  </article>
}

export default function FrequencyResultCard({ result, inputImage }) {
  return <div className="frequency-result">
    <div className="frequency-result-heading">
      <div><span className="eyebrow">{result.parameters.filter_family === 'gaussian' ? 'Gaussian' : 'Ideal'} Fourier filter</span><h2>{result.parameters.filter_type === 'low' ? 'Low-pass' : 'High-pass'} frequency analysis</h2></div>
      <span className="time-badge">{result.processing_time_ms} ms</span>
    </div>
    <div className="frequency-view-grid">
      <FrequencyView title="Input image" image={inputImage} description="The uploaded image used as the Fourier-domain input." />
      <FrequencyView title="Input image Fourier" image={result.fourier_input} description="Log-scaled magnitude of the uploaded image spectrum." />
      <FrequencyView title="Filter spatial kernel" image={result.filter_image} description="Signed inverse-Fourier response of the selected filter. Mid-gray represents zero; darker and lighter values represent negative and positive coefficients." />
      <FrequencyView title="Filter Fourier mask" image={result.fourier_filter} description="The transfer function H(u,v) used by the multiplication step. Sigma changes its width for Gaussian filters." />
      <FrequencyView title="Filtered result" image={result.image} description="Spatial image reconstructed after applying the Gaussian frequency filter." />
      <FrequencyView title="Filtered Fourier" image={result.fourier_result} description="The magnitude of F(u,v) × H(u,v), before the inverse Fourier transform." />
    </div>
  </div>
}
