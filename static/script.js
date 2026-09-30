const $ = (id) => document.getElementById(id);
const input = $('imageInput');
const dropzone = $('dropzone');
let selectedFile = null;
let currentPreviewUrl = null;
let currentResult = null;
const HISTORY_KEY = 'plantcare-ai-scan-history-v1';

function showError(message) { $('errorBox').textContent = message; $('errorBox').classList.remove('hidden'); }
function clearError() { $('errorBox').textContent = ''; $('errorBox').classList.add('hidden'); }
function setFile(file) {
  clearError();
  if (!file) return;
  if (!file.type.startsWith('image/')) { showError('Please choose an image file such as JPG, PNG, or WEBP.'); return; }
  if (file.size > 10 * 1024 * 1024) { showError('Please choose an image smaller than 10 MB.'); return; }
  selectedFile = file;
  if (currentPreviewUrl) URL.revokeObjectURL(currentPreviewUrl);
  currentPreviewUrl = URL.createObjectURL(file);
  $('preview').src = currentPreviewUrl;
  $('fileName').textContent = file.name;
  $('uploadPlaceholder').classList.add('hidden');
  $('previewWrap').classList.remove('hidden');
  $('analyzeBtn').disabled = false;
  $('resultBadge').textContent = 'Ready to analyze';
  $('resultBadge').className = 'result-badge neutral';
}
function resetFile() {
  selectedFile = null; input.value = '';
  if (currentPreviewUrl) URL.revokeObjectURL(currentPreviewUrl);
  currentPreviewUrl = null;
  $('preview').removeAttribute('src');
  $('previewWrap').classList.add('hidden'); $('uploadPlaceholder').classList.remove('hidden'); $('analyzeBtn').disabled = true;
}
$('chooseBtn').addEventListener('click', (e) => { e.stopPropagation(); input.click(); });
dropzone.addEventListener('click', (e) => { if (!e.target.closest('button')) input.click(); });
dropzone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
input.addEventListener('change', () => setFile(input.files[0]));
$('removeBtn').addEventListener('click', (e) => { e.stopPropagation(); resetFile(); });
['dragenter','dragover'].forEach(type => dropzone.addEventListener(type, e => { e.preventDefault(); dropzone.classList.add('dragover'); }));
['dragleave','drop'].forEach(type => dropzone.addEventListener(type, e => { e.preventDefault(); dropzone.classList.remove('dragover'); }));
dropzone.addEventListener('drop', e => setFile(e.dataTransfer.files[0]));

function renderResult(data) {
  currentResult = data;
  $('emptyResult').classList.add('hidden'); $('resultContent').classList.remove('hidden');
  $('predictionName').textContent = data.disease_type || data.prediction || 'Uncertain leaf condition';
  $('resultBadge').textContent = data.confidence < 35 ? 'Low confidence — verify' : 'AI screening result';
  $('resultBadge').className = `result-badge ${data.confidence < 35 ? 'warning' : 'neutral'}`;
  $('careSteps').innerHTML = '';
  (data.precautions || data.steps || []).forEach(step => { const li = document.createElement('li'); li.textContent = step; $('careSteps').appendChild(li); });
  $('solutionText').textContent = data.solution || 'Retake a clear photo and confirm the cause with a local agricultural expert before treatment.';
  $('limitationText').textContent = data.limitation || 'This is an AI screening result, not a confirmed diagnosis.';
  saveHistory(data);
}
$('analyzeBtn').addEventListener('click', async () => {
  if (!selectedFile) return;
  clearError(); $('analyzeBtn').disabled = true; $('progress').classList.remove('hidden');
  $('progressText').textContent = 'Checking the leaf for supported disease patterns and preparing precautions…';
  try {
    const form = new FormData(); form.append('image', selectedFile);
    const response = await fetch('/api/predict', { method: 'POST', body: form });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || 'The analysis failed. Please try again.');
    renderResult(data);
  } catch (error) {
    showError(`${error.message || 'Could not connect to the AI service.'} If this is the first run, check the terminal for model download or dependency errors.`);
  } finally { $('progress').classList.add('hidden'); $('analyzeBtn').disabled = !selectedFile; }
});
function getHistory() { try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); } catch { return []; } }
function saveHistory(data) {
  const entry = { prediction: data.prediction, confidence: data.confidence, date: new Date().toISOString(), fileName: selectedFile ? selectedFile.name : 'Leaf photo', preview: currentPreviewUrl || '' };
  const history = getHistory(); history.unshift(entry);
  // Store no image data in localStorage; this avoids filling browser storage.
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 12).map(({preview, ...rest}) => rest)));
  renderHistory();
}
function renderHistory() {
  const container = $('historyList'); const history = getHistory(); container.innerHTML = '';
  if (!history.length) { const empty = document.createElement('div'); empty.className = 'history-empty'; empty.textContent = 'No scans saved yet. Your completed scans will appear here.'; container.appendChild(empty); return; }
  history.forEach(item => {
    const row = document.createElement('div'); row.className = 'history-item';
    const thumb = document.createElement('div'); thumb.className = 'history-thumb'; thumb.style.display = 'grid'; thumb.style.placeItems = 'center'; thumb.style.color = '#4e8a4c'; thumb.textContent = '✳';
    const info = document.createElement('div'); info.className = 'history-info';
    const title = document.createElement('strong'); title.textContent = item.prediction;
    const sub = document.createElement('small'); sub.textContent = `${item.fileName} · ${new Date(item.date).toLocaleString()}`;
    info.append(title, sub); const confidence = document.createElement('span'); confidence.className = 'history-confidence'; confidence.textContent = `${item.confidence}%`;
    row.append(thumb, info, confidence); container.appendChild(row);
  });
}
$('clearHistory').addEventListener('click', () => { localStorage.removeItem(HISTORY_KEY); renderHistory(); });
renderHistory();
