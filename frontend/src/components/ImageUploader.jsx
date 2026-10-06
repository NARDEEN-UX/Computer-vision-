export default function ImageUploader({ file, onChange }) {
  return <label className="upload-box"><span className="upload-icon">＋</span><div><span className="eyebrow">Upload image</span><strong>{file ? file.name : 'Choose an image'}</strong><span className="muted">RGB or grayscale · PNG, JPG, or JPEG</span></div><input type="file" accept="image/*" onChange={(event) => onChange(event.target.files?.[0] ?? null)} /></label>
}
