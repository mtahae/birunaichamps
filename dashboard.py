"""
dashboard.py — BirunAI EKG Siniflandirma: Canli Egitim Dashboard
=================================================================

Flask tabanli web arayuzu. Egitim sirasinda http://localhost:5000
adresinden canli olarak izleme yapilabilir.

Ozellikler:
    - Loss ve F1 grafikleri (Chart.js, her 2 saniyede guncellenir)
    - Sinif bazli F1 metrikleri
    - Ilerleme cubugu + tahmini kalan sure
    - Early Stopping sayaci
    - Modern dark theme + glassmorphism
"""

import os
import sys
import json
import threading
from flask import Flask, jsonify, Response

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

app = Flask(__name__)

LOG_PATH = os.path.join(config.OUTPUT_DIR, "training_log.json")

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BirunAI — Eğitim Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  * { margin:0; padding:0; box-sizing:border-box; }

  body {
    font-family: 'Inter', sans-serif;
    background: #0a0a1a;
    color: #e0e0f0;
    min-height: 100vh;
    overflow-x: hidden;
  }

  body::before {
    content: '';
    position: fixed;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 20%, rgba(99,102,241,0.08) 0%, transparent 50%),
                radial-gradient(circle at 70% 80%, rgba(168,85,247,0.06) 0%, transparent 50%),
                radial-gradient(circle at 50% 50%, rgba(59,130,246,0.04) 0%, transparent 70%);
    z-index: -1;
    animation: bgPulse 20s ease-in-out infinite;
  }

  @keyframes bgPulse {
    0%, 100% { transform: rotate(0deg); }
    50% { transform: rotate(5deg); }
  }

  .header {
    padding: 24px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
    background: rgba(10,10,26,0.7);
    position: sticky; top: 0; z-index: 100;
  }

  .header h1 {
    font-size: 22px;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #a78bfa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .status-badge {
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.5px;
  }
  .status-training { background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
  .status-completed { background: rgba(99,102,241,0.15); color: #818cf8; border: 1px solid rgba(99,102,241,0.3); }
  .status-waiting { background: rgba(250,204,21,0.15); color: #fde047; border: 1px solid rgba(250,204,21,0.3); }

  .container { padding: 24px 32px; max-width: 1400px; margin: 0 auto; }

  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 24px; }

  .card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(8px);
    transition: all 0.3s ease;
  }
  .card:hover { border-color: rgba(129,140,248,0.2); background: rgba(255,255,255,0.05); }

  .metric-card { text-align: center; }
  .metric-value {
    font-size: 32px;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
  }
  .metric-label { font-size: 12px; color: #9ca3af; margin-top: 6px; text-transform: uppercase; letter-spacing: 1px; }

  .card-title {
    font-size: 14px;
    font-weight: 600;
    color: #a5b4fc;
    margin-bottom: 16px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
  }

  .progress-container { margin-bottom: 24px; }
  .progress-bar-bg {
    height: 8px;
    background: rgba(255,255,255,0.06);
    border-radius: 4px;
    overflow: hidden;
    margin-top: 10px;
  }
  .progress-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #6366f1, #a78bfa, #c084fc);
    transition: width 0.5s ease;
    position: relative;
  }
  .progress-bar-fill::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    animation: shimmer 2s infinite;
  }
  @keyframes shimmer { 0% { transform: translateX(-100%); } 100% { transform: translateX(100%); } }

  .progress-info { display: flex; justify-content: space-between; font-size: 13px; color: #9ca3af; }

  .class-bar-container { margin: 8px 0; }
  .class-bar-label { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px; }
  .class-bar-bg { height: 6px; background: rgba(255,255,255,0.06); border-radius: 3px; overflow: hidden; }
  .class-bar-fill { height: 100%; border-radius: 3px; transition: width 0.5s ease; }

  .es-indicator { display: flex; gap: 4px; margin-top: 10px; }
  .es-dot {
    width: 24px; height: 24px; border-radius: 6px;
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; font-weight: 600; color: #6b7280;
    transition: all 0.3s ease;
  }
  .es-dot.active { background: rgba(239,68,68,0.2); border-color: rgba(239,68,68,0.4); color: #f87171; }
  .es-dot.safe { background: rgba(34,197,94,0.2); border-color: rgba(34,197,94,0.4); color: #4ade80; }

  canvas { max-height: 280px; }

  .device-badge {
    padding: 4px 12px; border-radius: 8px;
    font-size: 11px; font-weight: 600;
    background: rgba(99,102,241,0.1); color: #818cf8;
    border: 1px solid rgba(99,102,241,0.2);
  }
</style>
</head>
<body>

<div class="header">
  <h1>⚡ BirunAI — EKG Eğitim Dashboard</h1>
  <div style="display:flex; align-items:center; gap:12px;">
    <span class="device-badge" id="deviceBadge">—</span>
    <span class="status-badge status-waiting" id="statusBadge">Bekleniyor</span>
  </div>
</div>

<div class="container">

  <!-- Metric Cards -->
  <div class="grid">
    <div class="card metric-card">
      <div class="metric-value" id="currentEpoch">—</div>
      <div class="metric-label">Epoch</div>
    </div>
    <div class="card metric-card">
      <div class="metric-value" id="bestF1">—</div>
      <div class="metric-label">En İyi F1</div>
    </div>
    <div class="card metric-card">
      <div class="metric-value" id="currentLR">—</div>
      <div class="metric-label">Learning Rate</div>
    </div>
    <div class="card metric-card">
      <div class="metric-value" id="eta">—</div>
      <div class="metric-label">Tahmini Kalan</div>
    </div>
  </div>

  <!-- Progress Bar -->
  <div class="card progress-container">
    <div class="progress-info">
      <span id="progressText">Eğitim bekleniyor...</span>
      <span id="progressPct">0%</span>
    </div>
    <div class="progress-bar-bg">
      <div class="progress-bar-fill" id="progressFill" style="width: 0%"></div>
    </div>
  </div>

  <!-- Charts -->
  <div class="grid-2">
    <div class="card">
      <div class="card-title">📉 Loss Eğrisi</div>
      <canvas id="lossChart"></canvas>
    </div>
    <div class="card">
      <div class="card-title">📈 Macro F1 Eğrisi</div>
      <canvas id="f1Chart"></canvas>
    </div>
  </div>

  <!-- Class F1 + Early Stopping -->
  <div class="grid-2">
    <div class="card">
      <div class="card-title">🎯 Sınıf Bazlı F1 (Validation)</div>
      <div class="class-bar-container">
        <div class="class-bar-label">
          <span>Normal</span>
          <span id="f1Class0">—</span>
        </div>
        <div class="class-bar-bg"><div class="class-bar-fill" id="f1Bar0" style="width:0%; background:#4ade80;"></div></div>
      </div>
      <div class="class-bar-container">
        <div class="class-bar-label">
          <span>AFIB</span>
          <span id="f1Class1">—</span>
        </div>
        <div class="class-bar-bg"><div class="class-bar-fill" id="f1Bar1" style="width:0%; background:#fb923c;"></div></div>
      </div>
      <div class="class-bar-container">
        <div class="class-bar-label">
          <span>AFL</span>
          <span id="f1Class2">—</span>
        </div>
        <div class="class-bar-bg"><div class="class-bar-fill" id="f1Bar2" style="width:0%; background:#60a5fa;"></div></div>
      </div>
      <div class="class-bar-container">
        <div class="class-bar-label">
          <span>LBBB</span>
          <span id="f1Class3">—</span>
        </div>
        <div class="class-bar-bg"><div class="class-bar-fill" id="f1Bar3" style="width:0%; background:#a78bfa;"></div></div>
      </div>
      <div class="class-bar-container">
        <div class="class-bar-label">
          <span>RBBB</span>
          <span id="f1Class4">—</span>
        </div>
        <div class="class-bar-bg"><div class="class-bar-fill" id="f1Bar4" style="width:0%; background:#f87171;"></div></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">⏱ Early Stopping Sayacı</div>
      <div class="es-indicator" id="esIndicator"></div>
      <p style="font-size:12px; color:#6b7280; margin-top:12px;" id="esText">
        Patience: 0 / 20
      </p>
    </div>
  </div>

</div>

<script>
const chartOptions = {
  responsive: true,
  maintainAspectRatio: true,
  animation: { duration: 400 },
  plugins: { legend: { labels: { color: '#9ca3af', font: { size: 11 } } } },
  scales: {
    x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#6b7280', font: { size: 10 } } },
    y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#6b7280', font: { size: 10 } } }
  }
};

const lossChart = new Chart(document.getElementById('lossChart'), {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      { label: 'Train Loss', data: [], borderColor: '#f87171', backgroundColor: 'rgba(248,113,113,0.1)', borderWidth: 2, pointRadius: 2, tension: 0.3, fill: true },
      { label: 'Val Loss', data: [], borderColor: '#60a5fa', backgroundColor: 'rgba(96,165,250,0.1)', borderWidth: 2, pointRadius: 2, tension: 0.3, fill: true }
    ]
  },
  options: chartOptions
});

const f1Chart = new Chart(document.getElementById('f1Chart'), {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      { label: 'Train F1', data: [], borderColor: '#a78bfa', borderWidth: 2, pointRadius: 2, tension: 0.3 },
      { label: 'Val Macro F1', data: [], borderColor: '#4ade80', backgroundColor: 'rgba(74,222,128,0.1)', borderWidth: 2, pointRadius: 3, tension: 0.3, fill: true }
    ]
  },
  options: chartOptions
});

// ES dots
const esContainer = document.getElementById('esIndicator');
for (let i = 0; i < 20; i++) {
  const dot = document.createElement('div');
  dot.className = 'es-dot';
  dot.textContent = i + 1;
  esContainer.appendChild(dot);
}

function updateDashboard(data) {
  if (!data || !data.epochs) return;

  // Status
  const badge = document.getElementById('statusBadge');
  badge.textContent = data.status === 'training' ? 'Eğitiliyor...' :
                      data.status === 'completed' ? 'Tamamlandı' : 'Bekleniyor';
  badge.className = 'status-badge status-' + (data.status === 'training' ? 'training' :
                     data.status === 'completed' ? 'completed' : 'waiting');

  // Device
  document.getElementById('deviceBadge').textContent = data.device || '—';

  const epochs = data.epochs;
  if (epochs.length === 0) return;

  const last = epochs[epochs.length - 1];

  // Metrics
  document.getElementById('currentEpoch').textContent = last.epoch + ' / ' + data.total_epochs;
  document.getElementById('bestF1').textContent = data.best_f1.toFixed(4);
  document.getElementById('currentLR').textContent = last.lr.toFixed(6);

  // ETA
  const avgDur = epochs.reduce((s, e) => s + e.duration_sec, 0) / epochs.length;
  const remaining = (data.total_epochs - last.epoch) * avgDur;
  const mins = Math.floor(remaining / 60);
  const secs = Math.floor(remaining % 60);
  document.getElementById('eta').textContent = mins + 'dk ' + secs + 's';

  // Progress
  const pct = (last.epoch / data.total_epochs * 100).toFixed(1);
  document.getElementById('progressFill').style.width = pct + '%';
  document.getElementById('progressPct').textContent = pct + '%';
  document.getElementById('progressText').textContent =
    'Epoch ' + last.epoch + ' / ' + data.total_epochs + ' | Son epoch: ' + last.duration_sec.toFixed(1) + 's';

  // Charts
  const labels = epochs.map(e => e.epoch);
  lossChart.data.labels = labels;
  lossChart.data.datasets[0].data = epochs.map(e => e.train_loss);
  lossChart.data.datasets[1].data = epochs.map(e => e.val_loss);
  lossChart.update('none');

  f1Chart.data.labels = labels;
  f1Chart.data.datasets[0].data = epochs.map(e => e.train_f1);
  f1Chart.data.datasets[1].data = epochs.map(e => e.val_f1_macro);
  f1Chart.update('none');

  // Class F1
  if (last.val_f1_class) {
    for (let i = 0; i < 5; i++) {
      const val = last.val_f1_class[i];
      document.getElementById('f1Class' + i).textContent = val.toFixed(4);
      document.getElementById('f1Bar' + i).style.width = (val * 100) + '%';
    }
  }

  // Early Stopping
  const dots = esContainer.children;
  for (let i = 0; i < 20; i++) {
    if (i < last.patience_counter) {
      dots[i].className = 'es-dot active';
    } else {
      dots[i].className = 'es-dot safe';
    }
  }
  document.getElementById('esText').textContent =
    'Patience: ' + last.patience_counter + ' / 20 — ' +
    (last.patience_counter >= 16 ? '⚠️ Durma yakın!' :
     last.patience_counter >= 10 ? '🟡 Dikkat' : '🟢 Stabil');
}

// 2 saniyede bir guncelle
async function fetchData() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    updateDashboard(data);
  } catch(e) {}
}

setInterval(fetchData, 2000);
fetchData();
</script>
</body>
</html>"""


@app.route('/')
def index():
    # Patience ve epoch degerlerini config'den oku, HTML icine enjekte et
    patience = config.EARLY_STOPPING_PATIENCE
    html = DASHBOARD_HTML
    html = html.replace('Patience: 0 / 20', f'Patience: 0 / {patience}')
    html = html.replace('for (let i = 0; i < 20; i++)', f'for (let i = 0; i < {patience}; i++)')
    html = html.replace(
        "const dots = esContainer.children;\n  for (let i = 0; i < 20; i++)",
        f"const dots = esContainer.children;\n  const MAX_PATIENCE = {patience};\n  const warn_high = Math.ceil(MAX_PATIENCE * 0.8);\n  const warn_mid  = Math.ceil(MAX_PATIENCE * 0.5);\n  for (let i = 0; i < MAX_PATIENCE; i++)"
    )
    html = html.replace(
        "'Patience: ' + last.patience_counter + ' / 20 \u2014 ' +\n    (last.patience_counter >= 16 ? '\u26a0\ufe0f Durma yak\u0131n!' :\n     last.patience_counter >= 10 ? '\U0001f7e1 Dikkat' : '\U0001f7e2 Stabil')",
        f"'Patience: ' + last.patience_counter + ' / {patience} \u2014 ' +\n    (last.patience_counter >= warn_high ? '\u26a0\ufe0f Durma yak\u0131n!' :\n     last.patience_counter >= warn_mid  ? '\U0001f7e1 Dikkat' : '\U0001f7e2 Stabil')"
    )
    return Response(html, mimetype='text/html')


@app.route('/api/status')
def api_status():
    try:
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return jsonify(data)
        else:
            return jsonify({"status": "waiting", "epochs": [], "total_epochs": 0})
    except Exception:
        return jsonify({"status": "waiting", "epochs": [], "total_epochs": 0})


def dashboard_baslat(port=5000):
    """Dashboard'u ayri thread'de baslat."""
    print(f"  Dashboard baslatildi: http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)


def dashboard_thread_baslat(port=5000):
    """Dashboard'u arka plan thread'inde baslat."""
    t = threading.Thread(target=dashboard_baslat, args=(port,), daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    print("=" * 70)
    print("BirunAI -- Dashboard (Canli Egitim Izleme)")
    print("=" * 70)
    print(f"\n  Tarayicida ac: http://localhost:5000")
    print(f"  Durdurmak icin: Ctrl+C\n")
    dashboard_baslat(5000)
