const canvas = document.getElementById('draw-canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const preview = document.getElementById('preprocessed-canvas');
const previewContext = preview.getContext('2d', { willReadFrequently: true });
let drawing = false;

// Setup drawing canvas
ctx.fillStyle = '#000000';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.strokeStyle = '#f4f8f2';
ctx.lineWidth = 22;

function position(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * canvas.width / rect.width,
    y: (event.clientY - rect.top) * canvas.height / rect.height
  };
}

canvas.addEventListener('pointerdown', e => {
  drawing = true;
  canvas.setPointerCapture(e.pointerId);
  const p = position(e);
  ctx.beginPath();
  ctx.moveTo(p.x, p.y);
  ctx.lineTo(p.x + 0.1, p.y + 0.1);
  ctx.stroke();
});

canvas.addEventListener('pointermove', e => {
  if (!drawing) return;
  const p = position(e);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  renderPreview(preprocess());
});

canvas.addEventListener('pointerup', () => {
  drawing = false;
  recognize();
});

canvas.addEventListener('pointerleave', () => {
  if (drawing) {
    drawing = false;
    recognize();
  }
});

function preprocess() {
  const source = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  let minX = canvas.width, minY = canvas.height, maxX = -1, maxY = -1;

  // Detect drawn ink bounds based on stroke brightness (light stroke on black background)
  for (let y = 0; y < canvas.height; y++) {
    for (let x = 0; x < canvas.width; x++) {
      const index = (y * canvas.width + x) * 4;
      const brightness = (source[index] + source[index + 1] + source[index + 2]) / 765;
      if (brightness > 0.08) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }
  }

  // If canvas is empty, return zeros
  if (maxX < 0 || maxY < 0) return Array(784).fill(0);

  const width = maxX - minX + 1;
  const height = maxY - minY + 1;

  // Scale into 20x20 bounding box inside 28x28 (MNIST standard normalization)
  const temp = document.createElement('canvas');
  temp.width = 28;
  temp.height = 28;
  const t = temp.getContext('2d');
  t.fillStyle = '#000000';
  t.fillRect(0, 0, 28, 28);
  t.imageSmoothingEnabled = true;
  t.imageSmoothingQuality = 'high';

  const targetSize = 20;
  const scale = targetSize / Math.max(width, height);
  const drawW = Math.max(1, Math.round(width * scale));
  const drawH = Math.max(1, Math.round(height * scale));
  const offsetX = (28 - drawW) / 2;
  const offsetY = (28 - drawH) / 2;

  t.drawImage(canvas, minX, minY, width, height, offsetX, offsetY, drawW, drawH);

  const raw = t.getImageData(0, 0, 28, 28).data;
  // Non-inverted grayscale pixel values: 0 for background, up to 1.0 for strokes
  const pixels = Array.from({ length: 784 }, (_, i) => raw[i * 4] / 255.0);

  // Center of mass calculation
  let mass = 0, sumX = 0, sumY = 0;
  for (let i = 0; i < 784; i++) {
    const value = pixels[i];
    if (value > 0.05) {
      mass += value;
      sumX += (i % 28) * value;
      sumY += Math.floor(i / 28) * value;
    }
  }

  if (mass === 0) return Array(784).fill(0);

  // Translate digit so its center of mass is centered at (13.5, 13.5)
  const shiftX = Math.round(13.5 - (sumX / mass));
  const shiftY = Math.round(13.5 - (sumY / mass));
  const clampedShiftX = Math.max(-5, Math.min(5, shiftX));
  const clampedShiftY = Math.max(-5, Math.min(5, shiftY));

  return Array.from({ length: 784 }, (_, i) => {
    const x = i % 28;
    const y = Math.floor(i / 28);
    const sourceX = x - clampedShiftX;
    const sourceY = y - clampedShiftY;
    if (sourceX >= 0 && sourceX < 28 && sourceY >= 0 && sourceY < 28) {
      return pixels[sourceY * 28 + sourceX];
    }
    return 0;
  });
}

function renderPreview(input) {
  const image = previewContext.createImageData(28, 28);
  input.forEach((value, index) => {
    const shade = Math.round(Math.min(1, Math.max(0, value)) * 255);
    image.data[index * 4] = shade;
    image.data[index * 4 + 1] = shade;
    image.data[index * 4 + 2] = shade;
    image.data[index * 4 + 3] = 255;
  });
  previewContext.putImageData(image, 0, 0);
}

function renderBars(probabilities) {
  const bars = document.getElementById('probability-bars');
  const maxVal = Math.max(...probabilities);
  bars.innerHTML = probabilities.map((value, index) =>
    `<div class="bar-row ${value === maxVal ? 'hot' : ''}">
      <span>${index}</span>
      <div class="bar-track">
        <div class="bar-fill" style="width:${Math.max(2, value * 100)}%"></div>
      </div>
      <span>${Math.round(value * 100)}%</span>
    </div>`
  ).join('');
}

async function recognize() {
  await DigitModel.ready;
  const input = preprocess();
  renderPreview(input);

  if (!input.some(value => value > 0.05)) return;

  const result = DigitModel.forward(input);
  const best = result.probabilities.indexOf(Math.max(...result.probabilities));

  document.getElementById('prediction-digit').textContent = best;
  document.getElementById('prediction-confidence').textContent = `${Math.round(result.probabilities[best] * 100)}% confidence`;
  document.getElementById('confidence-label').textContent = 'Signal complete';
  renderBars(result.probabilities);
  NetworkViz.update(result);
}

function clearCanvas() {
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  previewContext.fillStyle = '#000000';
  previewContext.fillRect(0, 0, 28, 28);
  document.getElementById('prediction-digit').textContent = '—';
  document.getElementById('prediction-confidence').textContent = 'Draw something to begin';
  document.getElementById('confidence-label').textContent = 'Awaiting input';
  document.getElementById('probability-bars').innerHTML = '';
  const zeroActivations = DigitModel.sizes.map((size, index) => Array(size).fill(index === DigitModel.sizes.length - 1 ? 0.1 : 0));
  NetworkViz.update({ activations: zeroActivations, weights: [] });
}

document.getElementById('predict-button').addEventListener('click', recognize);
document.getElementById('clear-button').addEventListener('click', clearCanvas);
document.getElementById('model-status').textContent = DigitModel.status;
document.getElementById('predict-button').disabled = true;