import axios from 'axios'

const client = axios.create({ baseURL: 'http://localhost:8000' })

export async function processImage(operation, file, parameters, secondaryFile = null) {
  const data = new FormData()
  if (operation.name === 'hybrid') {
    if (!file) throw new Error('Primary image is required for hybrid processing.')
    data.append('image_a', file)
    data.append('image_b', secondaryFile || file)
  } else {
    data.append('image', file)
  }
  Object.entries(parameters).forEach(([key, value]) => {
    if (value !== '' && value !== null && value !== undefined) data.append(key, String(value))
  })
  const response = await client.post(`/api/${operation.apiGroup || operation.group}/${operation.name}`, data)
  return response.data
}

export function dataUrlToFile(dataUrl, filename) {
  const [header, encoded] = dataUrl.split(',')
  const mime = header.match(/data:(.*?);base64/)?.[1] ?? 'image/png'
  const bytes = Uint8Array.from(atob(encoded), (character) => character.charCodeAt(0))
  return new File([bytes], filename, { type: mime })
}
