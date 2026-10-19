import axios from 'axios'

const client = axios.create({ baseURL: 'http://localhost:8000' })

export async function uploadImage(file) {
  if (!file) throw new Error('An image is required.')
  const data = new FormData()
  data.append('image', file)
  const response = await client.post('/api/images/upload', data)
  return response.data
}

export async function processImage(operation, file, parameters, secondaryFile = null, imageId = null, secondaryImageId = null) {
  const data = new FormData()
  if (operation.name === 'hybrid') {
    if (imageId) data.append('image_a_id', imageId)
    else if (file) data.append('image_a', file)
    else throw new Error('Primary image is required for hybrid processing.')
    if (secondaryImageId) data.append('image_b_id', secondaryImageId)
    else if (secondaryFile || file) data.append('image_b', secondaryFile || file)
    else throw new Error('Secondary image is required for hybrid processing.')
  } else if (imageId) {
    data.append('image_id', imageId)
  } else {
    if (!file) throw new Error('An image is required.')
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
