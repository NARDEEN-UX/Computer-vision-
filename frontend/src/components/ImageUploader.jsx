export default function ImageUploader({ file, onChange }) {
  return <label className="upload-box"><span className="upload-icon">＋</span><div><span className="eyebrow">Upload image</span><strong>{file ? file.name : 'Choose an RGB image'}</strong><span className="muted">PNG, JPG, or JPEG</span></div><input type="file" accept="image/png,image/jpeg,image/jpg" onChange={(event) => onChange(event.target.files?.[0] ?? null)} /></label>
}
