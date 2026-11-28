import geometry from '../assets/weddell/geometry.json'

const DEG = Math.PI / 180
const RADIUS = 6371228
const { width, height, left, top, step, meridian, sector, motion, ocean } = geometry
const [west, east, south, north] = sector
const sin = Math.sin(meridian * DEG)
const cos = Math.cos(meridian * DEG)

export const mapSize = { width, height }
export const assets = {
  satellite: new URL('../assets/weddell/satellite.webp', import.meta.url).href,
  mask: new URL('../assets/weddell/ice-mask.png', import.meta.url).href,
  concentration: new URL('../assets/weddell/concentration.webp', import.meta.url).href,
}
export const concentration = geometry.concentration

function viewPoint(x, y) {
  return [(x - left) / step, (top - y) / step]
}

function geoPoint(lat, lon) {
  const rho = 2 * RADIUS * Math.sin((90 + lat) * DEG / 2)
  return viewPoint(rho * Math.sin((lon - meridian) * DEG), rho * Math.cos((lon - meridian) * DEG))
}

function mapPoint(x, y) {
  return viewPoint(x * cos - y * sin, x * sin + y * cos)
}

export function cellPoint(row, col) {
  return mapPoint(motion.x0 + col * motion.dx, motion.y0 + row * motion.dy)
}

export function coordinates(row, col) {
  const x = motion.x0 + col * motion.dx
  const y = motion.y0 + row * motion.dy
  return {
    lat: 2 * Math.asin(Math.hypot(x, y) / (2 * RADIUS)) / DEG - 90,
    lon: Math.atan2(x, y) / DEG,
  }
}

export function cellAt(px, py) {
  const gx = left + px * step
  const gy = top - py * step
  const lon = Math.atan2(gx, gy) / DEG + meridian
  const lat = 2 * Math.asin(Math.hypot(gx, gy) / (2 * RADIUS)) / DEG - 90
  if (lon < west || lon > east || lat < south || lat > north) return null
  const row = Math.round((-gx * sin + gy * cos - motion.y0) / motion.dy)
  const col = Math.round((gx * cos + gy * sin - motion.x0) / motion.dx)
  return ocean[row]?.[col] === '1' ? { row, col } : null
}

const latitudes = [-60, -65, -70, -75, -80]
const longitudes = [-60, -40, -20, 0, 20]
const pointsAlongLatitude = (lat) => Array.from(
  { length: east - west + 1 }, (_, i) => geoPoint(lat, west + i),
)
const SPEED_COLORS = [[18, 83, 122], [22, 146, 180], [96, 233, 238], [255, 214, 108]]

function strokeLine(ctx, points) {
  ctx.beginPath()
  points.forEach(([x, y], index) => index ? ctx.lineTo(x, y) : ctx.moveTo(x, y))
  ctx.stroke()
}

function sectorPath() {
  const outline = new Path2D()
  const points = [...pointsAlongLatitude(north), ...pointsAlongLatitude(south).reverse()]
  outline.moveTo(...points[0])
  for (const point of points.slice(1)) outline.lineTo(...point)
  outline.closePath()
  return outline
}

function drawSpeed(ctx, field, mask, maxSpeed) {
  const layer = document.createElement('canvas')
  layer.width = width
  layer.height = height
  const ink = layer.getContext('2d')
  const colors = Array.from({ length: 64 }, (_, index) => {
    const position = (index / 63) * (SPEED_COLORS.length - 1)
    const low = Math.min(SPEED_COLORS.length - 2, Math.floor(position))
    const fraction = position - low
    const rgb = SPEED_COLORS[low].map(
      (color, band) => Math.round(color * (1 - fraction) + SPEED_COLORS[low + 1][band] * fraction),
    )
    return `rgb(${rgb.join(',')})`
  })
  const [ox, oy] = cellPoint(0, 0)
  ink.setTransform(cos / step, -sin / step, -sin / step, -cos / step, ox, oy)
  field.speed.forEach((line, row) => line.forEach((speed, col) => {
    if (speed == null || !Number.isFinite(speed)) return
    ink.fillStyle = colors[Math.max(0, Math.min(63, Math.round(63 * speed / maxSpeed)))]
    ink.fillRect((col - .5) * motion.dx, (row - .5) * motion.dy, motion.dx, motion.dy)
  }))
  ink.resetTransform()
  ink.globalCompositeOperation = 'destination-out'
  ink.drawImage(mask, 0, 0)
  ctx.globalAlpha = .76
  ctx.drawImage(layer, 0, 0)
  ctx.globalAlpha = 1
}

function drawGrid(ctx, unit) {
  ctx.save()
  ctx.strokeStyle = '#dcefff38'
  ctx.lineWidth = .65 * unit
  ctx.setLineDash([2 * unit, 3 * unit])
  for (const lat of latitudes) strokeLine(ctx, pointsAlongLatitude(lat))
  for (const lon of longitudes) strokeLine(ctx, [geoPoint(north, lon), geoPoint(south, lon)])
  ctx.restore()
}

function label(ctx, text, x, y, unit, { size = 10, color = '#d7edfa', alignment = 'center' } = {}) {
  ctx.font = `600 ${size * unit}px system-ui, sans-serif`
  ctx.textAlign = alignment
  ctx.textBaseline = 'middle'
  ctx.lineJoin = 'round'
  ctx.lineWidth = 2.5 * unit
  ctx.strokeStyle = '#081422e8'
  ctx.strokeText(text, x, y)
  ctx.fillStyle = color
  ctx.fillText(text, x, y)
}

function drawLabels(ctx, unit) {
  for (const lat of latitudes) {
    const [x, y] = geoPoint(lat, west)
    label(ctx, `${-lat}°S`, x - 6 * unit, y + 10 * unit, unit, { size: 9, alignment: 'right' })
  }
  for (const lon of longitudes) {
    const [x, y] = geoPoint(north, lon)
    label(ctx, lon === 0 ? '0°' : `${Math.abs(lon)}°${lon < 0 ? 'W' : 'E'}`, x, y - 8 * unit, unit, { size: 9 })
  }
  label(ctx, '威德尔海', ...geoPoint(-69.3, -33), unit, { size: 12 })
  label(ctx, '南极半岛', ...geoPoint(-67.6, -65.4), unit)
  label(ctx, '菲尔希纳–龙尼冰架', ...geoPoint(-79.3, -54), unit)
  label(ctx, '布伦特冰架', ...geoPoint(-74.8, -25.5), unit)
}

function drawArrows(ctx, vectors, unit) {
  if (!vectors?.rows) return
  ctx.lineCap = 'round'
  for (const [index, row] of vectors.rows.entries()) {
    const col = vectors.cols[index]
    if (ocean[row]?.[col] !== '1') continue
    const u = vectors.u[index]
    const v = vectors.v[index]
    const speed = Math.hypot(u, v)
    if (speed < .2) continue
    const dx = (u * cos - v * sin) / speed
    const dy = -(u * sin + v * cos) / speed
    const length = (3.2 + 6.8 * Math.min(speed / 28, 1)) * unit
    const [x, y] = cellPoint(row, col)
    const tipX = x + dx * length * .5
    const tipY = y + dy * length * .5
    const head = Math.min(3.0 * unit, length * .42)
    ctx.beginPath()
    ctx.moveTo(x - dx * length * .5, y - dy * length * .5)
    ctx.lineTo(tipX, tipY)
    ctx.moveTo(tipX - dx * head - dy * head * .58, tipY - dy * head + dx * head * .58)
    ctx.lineTo(tipX, tipY)
    ctx.lineTo(tipX - dx * head + dy * head * .58, tipY - dy * head - dx * head * .58)
    ctx.strokeStyle = '#081320bb'
    ctx.lineWidth = 2.3 * unit
    ctx.stroke()
    ctx.strokeStyle = '#f3faff'
    ctx.lineWidth = .9 * unit
    ctx.stroke()
  }
}

function marker(ctx, point, size, color, unit) {
  ctx.beginPath()
  ctx.arc(...point, size * unit, 0, 2 * Math.PI)
  ctx.fillStyle = '#122234'
  ctx.fill()
  ctx.strokeStyle = color
  ctx.lineWidth = 1.5 * unit
  ctx.stroke()
}

function drawSelection(ctx, selected, path, unit) {
  if (ocean[selected.row]?.[selected.col] !== '1') return
  marker(ctx, cellPoint(selected.row, selected.col), 3.9, '#fde18d', unit)
  if (!path?.col?.length) return
  const points = path.col.map((col, index) => cellPoint(path.row[index], col))
  const end = points.at(-1)
  ctx.lineWidth = 4.4 * unit
  ctx.strokeStyle = '#081220'
  strokeLine(ctx, points)
  ctx.lineWidth = 2.3 * unit
  ctx.strokeStyle = '#fcd34d'
  strokeLine(ctx, points)
  marker(ctx, end, 3, '#fcd34d', unit)
  label(ctx, `+${path.hours.at(-1)}h`, end[0] + 10 * unit, end[1] - 6 * unit, unit, {
    size: 9, color: '#ffe5a9', alignment: 'left',
  })
}

export function drawWeddell(ctx, { images, field, layer, maximum, selected, path, unit }) {
  const border = sectorPath()
  ctx.save()
  ctx.clip(border)
  ctx.drawImage(images.satellite, 0, 0)
  if (layer === 'concentration') ctx.drawImage(images.concentration, 0, 0)
  if (layer === 'speed' && field) drawSpeed(ctx, field, images.mask, maximum)
  drawGrid(ctx, unit)
  drawArrows(ctx, field?.vectors, unit)
  drawSelection(ctx, selected, path, unit)
  ctx.restore()
  ctx.lineWidth = 1 * unit
  ctx.strokeStyle = '#86afcc99'
  ctx.stroke(border)
  drawLabels(ctx, unit)
}
