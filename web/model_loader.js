const DigitModel = (() => {
  let sizes = [784, 16, 16, 10];
  let weights = null;
  let biases = null;

  const relu = value => Math.max(0, value);

  const softmax = values => {
    const max = Math.max(...values);
    const exps = values.map(value => Math.exp(value - max));
    const total = exps.reduce((a, b) => a + b, 0);
    return exps.map(value => value / total);
  };

  const forward = input => {
    if (!weights) throw new Error('Model weights are still loading');
    let activation = input;
    const activations = [input];

    for (let layer = 0; layer < weights.length; layer += 1) {
      activation = weights[layer].map((row, index) =>
        row.reduce((sum, value, i) => sum + value * activation[i], biases[layer]?.[index] || 0)
      );
      activation = layer === weights.length - 1 ? softmax(activation) : activation.map(relu);
      activations.push(activation);
    }

    return {
      probabilities: activations[activations.length - 1],
      activations,
      weights
    };
  };

  const ready = fetch('model_weights.json')
    .then(response => {
      if (!response.ok) throw new Error(`HTTP ${response.status}: Failed to load model weights`);
      return response.json();
    })
    .then(model => {
      sizes = model.sizes;
      weights = model.weights;
      biases = model.biases;
      const hidden = sizes.slice(1, -1).join(' × ');
      const statusEl = document.getElementById('model-status');
      if (statusEl) statusEl.textContent = `MNIST network ready · ${hidden} hidden layers`;
      const btn = document.getElementById('predict-button');
      if (btn) btn.disabled = false;
    })
    .catch(err => {
      console.error(err);
      const statusEl = document.getElementById('model-status');
      if (statusEl) statusEl.textContent = 'Error loading model. Run python main.py to serve locally.';
    });

  return { sizes, forward, ready, status: 'Loading trained MNIST network...' };
})();