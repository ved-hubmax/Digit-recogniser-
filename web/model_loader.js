const DigitModel = (() => {
  const sizes = [784, 16, 16, 10];
  let weights = null;
  let biases = null;
  const relu = value => Math.max(0, value);
  const softmax = values => { const max = Math.max(...values); const exps = values.map(value => Math.exp(value - max)); const total = exps.reduce((a,b) => a + b, 0); return exps.map(value => value / total); };
  const forward = input => {
    if (!weights) throw new Error('Model weights are still loading');
    let activation = input; const activations = [input];
    for (let layer = 0; layer < weights.length; layer += 1) {
      activation = weights[layer].map((row, index) => row.reduce((sum, value, i) => sum + value * activation[i], biases[layer]?.[index] || 0));
      activation = layer === weights.length - 1 ? softmax(activation) : activation.map(relu); activations.push(activation);
    }
    return { probabilities: activations[3], activations, weights };
  };
  const ready = fetch('model_weights.json').then(response => response.json()).then(model => { weights = model.weights; biases = model.biases; document.getElementById('model-status').textContent = 'MNIST network ready · 95.6% test accuracy'; document.getElementById('predict-button').disabled = false; });
  return { sizes, forward, ready, status: 'Loading trained MNIST network...' };
})();