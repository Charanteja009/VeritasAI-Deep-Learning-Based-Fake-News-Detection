document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const form = document.getElementById('predictionForm');
  const newsTitle = document.getElementById('newsTitle');
  const newsText = document.getElementById('newsText');
  const textStats = document.getElementById('textStats');
  const clearBtn = document.getElementById('clearBtn');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const btnSpinner = document.getElementById('btnSpinner');
  const btnLabel = document.getElementById('btnLabel');
  const btnArrow = document.getElementById('btnArrow');

  const emptyState = document.getElementById('emptyState');
  const resultsContent = document.getElementById('resultsContent');
  const modelStatusBadge = document.getElementById('modelStatusBadge');

  const verdictCard = document.getElementById('verdictCard');
  const verdictKicker = document.getElementById('verdictKicker');
  const verdictTitle = document.getElementById('verdictTitle');
  const verdictDesc = document.getElementById('verdictDesc');
  const confidenceValue = document.getElementById('confidenceValue');

  const readoutReal = document.getElementById('readoutReal');
  const readoutFake = document.getElementById('readoutFake');
  const probRealBar = document.getElementById('probRealBar');
  const probFakeBar = document.getElementById('probFakeBar');

  const resWordCount = document.getElementById('resWordCount');
  const resCharCount = document.getElementById('resCharCount');
  const resLatency = document.getElementById('resLatency');

  const realChips = document.getElementById('realChips');
  const fakeChips = document.getElementById('fakeChips');
  const cleanedPreview = document.getElementById('cleanedPreview');

  const samplesContainer = document.getElementById('samplesContainer');
  const toast = document.getElementById('toast');

  let sampleArticles = [];

  // Live text tally update
  function updateCounters() {
    const titleVal = newsTitle.value || '';
    const textVal = newsText.value || '';
    const combined = `${titleVal} ${textVal}`.trim();
    const words = combined ? combined.split(/\s+/).filter(Boolean).length : 0;
    const chars = combined.length;
    textStats.textContent = `${words} words • ${chars} chars`;
  }

  newsTitle.addEventListener('input', updateCounters);
  newsText.addEventListener('input', updateCounters);

  // Form Reset
  clearBtn.addEventListener('click', () => {
    newsTitle.value = '';
    newsText.value = '';
    updateCounters();
    emptyState.classList.remove('hidden');
    resultsContent.classList.add('hidden');
    modelStatusBadge.textContent = 'Awaiting Input';
    modelStatusBadge.style.color = '';
    newsText.focus();
  });

  // Minimalist Toast Helper
  function showToast(message, isError = false) {
    toast.textContent = message;
    toast.style.borderLeftColor = isError ? 'var(--color-coral)' : 'var(--color-stone-muted)';
    toast.classList.add('show');
    setTimeout(() => {
      toast.classList.remove('show');
    }, 3800);
  }

  // Fetch samples from backend API
  async function loadSamples() {
    try {
      const res = await fetch('/api/samples');
      if (!res.ok) return;
      sampleArticles = await res.json();
      renderSampleButtons();
    } catch (err) {
      console.warn('API samples not available, using embedded benchmarks.', err);
    }
  }

  function renderSampleButtons() {
    if (!sampleArticles || sampleArticles.length === 0) return;
    samplesContainer.innerHTML = '';
    sampleArticles.forEach((sample) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      const isFake = sample.category === 'fake';
      btn.className = `sample-tag ${isFake ? 'sample-tag-fake' : 'sample-tag-real'}`;
      btn.textContent = `${isFake ? 'Fake' : 'Real'} • ${sample.title.slice(0, 26)}…`;
      btn.title = `${sample.title}\n(${sample.description})`;
      btn.addEventListener('click', () => {
        newsTitle.value = sample.title;
        newsText.value = sample.text;
        updateCounters();
        executePrediction();
      });
      samplesContainer.appendChild(btn);
    });
  }

  // Prediction Runner
  async function executePrediction() {
    const title = newsTitle.value.trim();
    const text = newsText.value.trim();

    if ((title.length + text.length) < 10) {
      showToast('Please enter at least 10 characters to evaluate.', true);
      newsText.focus();
      return;
    }

    setLoading(true);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, text })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Analysis could not be completed.');
      }

      const data = await response.json();
      displayResults(data);
    } catch (error) {
      showToast(error.message || 'Error communicating with prediction server.', true);
    } finally {
      setLoading(false);
    }
  }

  function setLoading(isLoading) {
    if (isLoading) {
      analyzeBtn.disabled = true;
      btnSpinner.style.display = 'inline-block';
      btnArrow.style.display = 'none';
      btnLabel.textContent = 'Evaluating…';
      modelStatusBadge.textContent = 'Inference in Progress';
    } else {
      analyzeBtn.disabled = false;
      btnSpinner.style.display = 'none';
      btnArrow.style.display = 'inline';
      btnLabel.textContent = 'Evaluate Authenticity';
    }
  }

  // Display Assessment Results
  function displayResults(data) {
    emptyState.classList.add('hidden');
    resultsContent.classList.remove('hidden');

    const isFake = data.is_fake;

    // Status Badge
    modelStatusBadge.textContent = isFake ? 'Flagged as Fabricated' : 'Verified as Authentic';
    modelStatusBadge.style.color = isFake ? 'var(--color-coral)' : 'var(--color-black)';

    // Verdict Card
    verdictCard.className = `verdict-card ${isFake ? 'fake' : 'authentic'}`;
    verdictKicker.textContent = isFake ? 'Anomaly Detected' : 'Verified Provenance';
    verdictTitle.textContent = isFake ? 'Fabricated / Fake Claim' : 'Authentic Journalism';
    verdictDesc.textContent = isFake
      ? 'Vocabulary and framing strongly correlate with fabricated or sensationalized narratives.'
      : 'Syntactic composition aligns with verified wire-service reporting and attributed sources.';

    confidenceValue.textContent = `${data.confidence.toFixed(1)}%`;

    // Probability Split
    const realPct = data.probabilities.real;
    const fakePct = data.probabilities.fake;

    readoutReal.textContent = `Real ${realPct.toFixed(1)}%`;
    readoutFake.textContent = `Fake ${fakePct.toFixed(1)}%`;

    requestAnimationFrame(() => {
      probRealBar.style.width = `${realPct}%`;
      probFakeBar.style.width = `${fakePct}%`;
    });

    // Telemetry
    resWordCount.textContent = data.metrics.word_count.toLocaleString();
    resCharCount.textContent = data.metrics.char_count.toLocaleString();
    resLatency.textContent = `${data.metrics.latency_ms} ms`;

    // Linguistic Tokens
    renderTokens(realChips, data.explainability.real_signals, 'token-pill-real');
    renderTokens(fakeChips, data.explainability.fake_signals, 'token-pill-fake');

    // Cleaned Preview
    cleanedPreview.textContent = data.cleaned_preview || '(None)';
  }

  function renderTokens(container, signals, variantClass) {
    container.innerHTML = '';
    if (!signals || signals.length === 0) {
      container.innerHTML = '<span class="token-empty">No active n-gram tokens</span>';
      return;
    }
    signals.forEach(sig => {
      const pill = document.createElement('span');
      pill.className = `token-pill ${variantClass}`;
      pill.innerHTML = `${escapeHtml(sig.word)} <span class="token-weight">+${sig.weight}</span>`;
      container.appendChild(pill);
    });
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // Event handlers
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    executePrediction();
  });

  // Init
  updateCounters();
  loadSamples();
});
