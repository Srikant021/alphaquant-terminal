<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Nifty 50 — Expected Move</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

  :root {
    --bg:        #050a0f;
    --panel:     #0a1520;
    --border:    #0e2a3a;
    --cyan:      #00e5ff;
    --green:     #00ff88;
    --red:       #ff3355;
    --white:     #e8f4f8;
    --muted:     #4a6a7a;
    --gold:      #ffd54f;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--white);
    font-family: 'Rajdhani', sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 24px 16px;
    overflow-x: hidden;
  }

  /* scanline overlay */
  body::before {
    content: '';
    position: fixed; inset: 0;
    background: repeating-linear-gradient(
      0deg,
      transparent,
      transparent 2px,
      rgba(0,229,255,0.015) 2px,
      rgba(0,229,255,0.015) 4px
    );
    pointer-events: none;
    z-index: 999;
  }

  header {
    width: 100%; max-width: 960px;
    display: flex; align-items: baseline; gap: 16px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 12px;
    margin-bottom: 24px;
  }
  header h1 {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.5rem;
    color: var(--cyan);
    letter-spacing: 3px;
    text-transform: uppercase;
  }
  header span {
    font-size: 0.8rem;
    color: var(--muted);
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 1px;
  }
  .blink { animation: blink 1.2s step-end infinite; }
  @keyframes blink { 50% { opacity: 0; } }

  .grid {
    width: 100%; max-width: 960px;
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 12px;
    margin-bottom: 20px;
  }

  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 16px 20px;
    position: relative;
    overflow: hidden;
  }
  .card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
  }
  .card.c-cyan::after  { background: var(--cyan); }
  .card.c-green::after { background: var(--green); }
  .card.c-red::after   { background: var(--red); }
  .card.c-gold::after  { background: var(--gold); }

  .card .label {
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
  }
  .card .value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
  }
  .card .sub {
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 4px;
    font-family: 'Share Tech Mono', monospace;
  }
  .card.c-cyan  .value { color: var(--cyan); }
  .card.c-green .value { color: var(--green); }
  .card.c-red   .value { color: var(--red); }
  .card.c-gold  .value { color: var(--gold); }

  .strike-row {
    width: 100%; max-width: 960px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 20px;
  }
  .strike-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .strike-icon { font-size: 1.8rem; }
  .strike-info .label {
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
  }
  .strike-info .val {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.4rem;
  }
  .strike-info .val.green { color: var(--green); }
  .strike-info .val.red   { color: var(--red); }
  .strike-info .hint {
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 2px;
  }

  .chart-wrapper {
    width: 100%; max-width: 960px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 20px;
    margin-bottom: 20px;
  }
  .chart-title {
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 14px;
  }
  canvas { width: 100% !important; }

  .math-box {
    width: 100%; max-width: 960px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 16px 20px;
    margin-bottom: 20px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    color: var(--muted);
    line-height: 1.8;
  }
  .math-box span { color: var(--cyan); }

  #status {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    color: var(--muted);
    margin-bottom: 20px;
    letter-spacing: 1px;
  }
  #status.error { color: var(--red); }

  .refresh-btn {
    background: transparent;
    border: 1px solid var(--cyan);
    color: var(--cyan);
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 2px;
    padding: 8px 20px;
    cursor: pointer;
    border-radius: 2px;
    margin-bottom: 24px;
    transition: background 0.2s, color 0.2s;
  }
  .refresh-btn:hover {
    background: var(--cyan);
    color: var(--bg);
  }

  .skeleton {
    background: linear-gradient(90deg, var(--border) 25%, #0d2235 50%, var(--border) 75%);
    background-size: 200% 100%;
    animation: shimmer 1.4s infinite;
    border-radius: 3px;
    height: 1.5rem;
    width: 80%;
  }
  @keyframes shimmer { to { background-position: -200% 0; } }

  footer {
    font-size: 0.7rem;
    color: var(--muted);
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 1px;
    text-align: center;
  }

  @media (max-width: 620px) {
    .grid { grid-template-columns: 1fr 1fr; }
    .strike-row { grid-template-columns: 1fr; }
    header h1 { font-size: 1.1rem; }
  }
</style>
</head>
<body>

<header>
  <h1>NIFTY 50 ▸ EXPECTED MOVE</h1>
  <span id="ts">LOADING<span class="blink">_</span></span>
</header>

<div id="status">◎ FETCHING MARKET DATA...</div>

<div class="grid">
  <div class="card c-cyan">
    <div class="label">Nifty Spot</div>
    <div class="value" id="v-spot"><div class="skeleton"></div></div>
    <div class="sub" id="v-spot-chg">—</div>
  </div>
  <div class="card c-gold">
    <div class="label">India VIX</div>
    <div class="value" id="v-vix"><div class="skeleton"></div></div>
    <div class="sub" id="v-vix-chg">—</div>
  </div>
  <div class="card c-cyan">
    <div class="label">Daily Implied Move</div>
    <div class="value" id="v-move"><div class="skeleton"></div></div>
    <div class="sub" id="v-move-pct">—</div>
  </div>
</div>

<div class="strike-row">
  <div class="strike-card">
    <div class="strike-icon">🟢</div>
    <div class="strike-info">
      <div class="label">Safe Short CALL above</div>
      <div class="val green" id="v-upper">—</div>
      <div class="hint">+1 SD upper bound</div>
    </div>
  </div>
  <div class="strike-card">
    <div class="strike-icon">🔴</div>
    <div class="strike-info">
      <div class="label">Safe Short PUT below</div>
      <div class="val red" id="v-lower">—</div>
      <div class="hint">−1 SD lower bound</div>
    </div>
  </div>
</div>

<div class="chart-wrapper">
  <div class="chart-title">◈ 15-DAY CLOSE + TOMORROW'S IMPLIED RANGE</div>
  <canvas id="chart" height="320"></canvas>
</div>

<div class="math-box" id="math-box">
  <span>FORMULA:</span> Daily Vol = (VIX / 100) × √(1/365) &nbsp;|&nbsp; Expected Move = Spot × Daily Vol
</div>

<button class="refresh-btn" onclick="init()">⟳ REFRESH DATA</button>

<footer>Data via Yahoo Finance public API &nbsp;·&nbsp; For educational purposes only &nbsp;·&nbsp; Not financial advice</footer>

<script>
let chartInst = null;

async function fetchYahoo(symbol, range = '1mo', interval = '1d') {
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?range=${range}&interval=${interval}&events=history`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${symbol}`);
  const json = await res.json();
  const result = json.chart.result[0];
  const timestamps = result.timestamp.map(t => new Date(t * 1000));
  const closes = result.indicators.quote[0].close;
  return { timestamps, closes };
}

function setEl(id, html) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = html;
}

function fmt(n, dec = 2) {
  return n.toLocaleString('en-IN', { minimumFractionDigits: dec, maximumFractionDigits: dec });
}

function roundStrike(price) {
  return Math.round(price / 50) * 50;
}

function buildChart(labels, closes, spot, upper, lower, vix, movePts) {
  const ctx = document.getElementById('chart').getContext('2d');
  if (chartInst) chartInst.destroy();

  // Readout box plugin — mirrors the Python ax.text() data box
  const readoutBoxPlugin = {
    id: 'readoutBox',
    afterDraw(chart) {
      const { ctx: c, chartArea: { left, top, bottom } } = chart;
      const lines = [
        { label: '⚡ Current VIX',           value: fmt(vix, 2),               color: '#ffd54f' },
        { label: '🎯 Spot Price',             value: fmt(spot, 2),              color: '#e8f4f8' },
        { label: '📏 Expected Move',          value: `± ${fmt(movePts, 1)} pts`, color: '#00e5ff' },
        { label: '──────────────',            value: '',                         color: '#4a6a7a' },
        { label: '🟢 Safe Call Strike >',     value: String(roundStrike(upper)), color: '#00ff88' },
        { label: '🔴 Safe Put Strike  <',     value: String(roundStrike(lower)), color: '#ff3355' },
      ];

      const pad = 12, lineH = 22, fontSize = 12;
      const boxW = 270, boxH = lines.length * lineH + pad * 2;
      const bx = left + 10, by = top + (bottom - top) * 0.08;

      // Box background + border
      c.save();
      c.globalAlpha = 0.88;
      c.fillStyle = '#050a0f';
      c.strokeStyle = '#e8f4f8';
      c.lineWidth = 1.2;
      const r = 6;
      c.beginPath();
      c.moveTo(bx + r, by); c.lineTo(bx + boxW - r, by);
      c.quadraticCurveTo(bx + boxW, by, bx + boxW, by + r);
      c.lineTo(bx + boxW, by + boxH - r);
      c.quadraticCurveTo(bx + boxW, by + boxH, bx + boxW - r, by + boxH);
      c.lineTo(bx + r, by + boxH);
      c.quadraticCurveTo(bx, by + boxH, bx, by + boxH - r);
      c.lineTo(bx, by + r);
      c.quadraticCurveTo(bx, by, bx + r, by);
      c.closePath();
      c.fill();
      c.globalAlpha = 1;
      c.stroke();

      // Text rows
      c.font = `bold ${fontSize}px "Share Tech Mono", monospace`;
      lines.forEach((ln, i) => {
        const y = by + pad + fontSize + i * lineH;
        c.fillStyle = ln.color;
        c.fillText(ln.label, bx + pad, y);
        if (ln.value) {
          c.textAlign = 'right';
          c.fillStyle = ln.color;
          c.fillText(ln.value, bx + boxW - pad, y);
          c.textAlign = 'left';
        }
      });
      c.restore();
    }
  };

  const allLabels = [...labels, 'Tomorrow'];
  const mainData  = [...closes, null];
  const spotLine  = [...closes.map(() => null), spot];
  const upperLine = [...closes.map(() => null), upper];
  const lowerLine = [...closes.map(() => null), lower];

  // cone fill dataset
  const coneUpper = closes.map(() => null);
  coneUpper[coneUpper.length - 1] = upper;
  coneUpper.push(upper);

  chartInst = new Chart(ctx, {
    plugins: [readoutBoxPlugin],
    type: 'line',
    data: {
      labels: allLabels,
      datasets: [
        {
          label: 'Nifty 50 Close',
          data: mainData,
          borderColor: '#00e5ff',
          backgroundColor: 'rgba(0,229,255,0.06)',
          borderWidth: 2,
          pointRadius: 3,
          pointBackgroundColor: '#00e5ff',
          tension: 0.3,
          fill: false,
          spanGaps: false,
        },
        {
          label: 'Spot (carry-fwd)',
          data: (() => {
            const d = closes.map(() => null);
            d[d.length-1] = spot;
            d.push(spot);
            return d;
          })(),
          borderColor: 'rgba(255,255,255,0.5)',
          borderWidth: 1.5,
          borderDash: [4,4],
          pointRadius: [0,0],
          fill: false,
          tension: 0,
        },
        {
          label: 'Upper Bound (+1 SD)',
          data: (() => {
            const d = closes.map(() => null);
            d[d.length-1] = spot;
            d.push(upper);
            return d;
          })(),
          borderColor: '#00ff88',
          borderWidth: 2,
          borderDash: [6,3],
          pointRadius: [0,6],
          pointBackgroundColor: '#00ff88',
          fill: false,
          tension: 0,
        },
        {
          label: 'Lower Bound (-1 SD)',
          data: (() => {
            const d = closes.map(() => null);
            d[d.length-1] = spot;
            d.push(lower);
            return d;
          })(),
          borderColor: '#ff3355',
          borderWidth: 2,
          borderDash: [6,3],
          pointRadius: [0,6],
          pointBackgroundColor: '#ff3355',
          fill: false,
          tension: 0,
        },
      ]
    },
    options: {
      responsive: true,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          labels: {
            color: '#4a6a7a',
            font: { family: 'Share Tech Mono', size: 11 },
            boxWidth: 16,
          }
        },
        tooltip: {
          backgroundColor: '#0a1520',
          borderColor: '#0e2a3a',
          borderWidth: 1,
          titleColor: '#00e5ff',
          bodyColor: '#e8f4f8',
          titleFont: { family: 'Share Tech Mono' },
          bodyFont: { family: 'Share Tech Mono', size: 12 },
          callbacks: {
            label: ctx => {
              if (ctx.raw == null) return null;
              return ` ${ctx.dataset.label}: ${fmt(ctx.raw)}`;
            }
          }
        }
      },
      scales: {
        x: {
          ticks: {
            color: '#4a6a7a',
            font: { family: 'Share Tech Mono', size: 10 },
            maxTicksLimit: 8,
            maxRotation: 0,
          },
          grid: { color: '#0d1e2a' }
        },
        y: {
          ticks: {
            color: '#4a6a7a',
            font: { family: 'Share Tech Mono', size: 10 },
            callback: v => fmt(v, 0),
          },
          grid: { color: '#0d1e2a' }
        }
      }
    }
  });
}

async function init() {
  setEl('status', '◎ FETCHING MARKET DATA...');
  document.getElementById('status').className = '';

  try {
    const [nifty, vix] = await Promise.all([
      fetchYahoo('^NSEI', '1mo', '1d'),
      fetchYahoo('^INDIAVIX', '5d', '1d'),
    ]);

    // Filter nulls
    const validNifty = nifty.closes.map((c, i) => ({ c, t: nifty.timestamps[i] })).filter(x => x.c != null);
    const validVix   = vix.closes.filter(c => c != null);

    if (!validNifty.length || !validVix.length) throw new Error('Empty data returned');

    const spot        = validNifty[validNifty.length - 1].c;
    const currentVix  = validVix[validVix.length - 1];
    const dailyVol    = (currentVix / 100) * Math.sqrt(1 / 365);
    const movePts     = spot * dailyVol;
    const movePct     = dailyVol * 100;
    const upper       = spot + movePts;
    const lower       = spot - movePts;

    // Previous values for change display
    const prevSpot = validNifty.length > 1 ? validNifty[validNifty.length - 2].c : spot;
    const prevVix  = validVix.length > 1 ? validVix[validVix.length - 2] : currentVix;

    const spotChg = spot - prevSpot;
    const vixChg  = currentVix - prevVix;

    // Update cards
    setEl('v-spot', fmt(spot, 2));
    setEl('v-spot-chg', `${spotChg >= 0 ? '▲' : '▼'} ${fmt(Math.abs(spotChg), 2)} (${fmt(Math.abs(spotChg/prevSpot*100), 2)}%)`);

    setEl('v-vix', fmt(currentVix, 2));
    setEl('v-vix-chg', `${vixChg >= 0 ? '▲' : '▼'} ${fmt(Math.abs(vixChg), 2)} pts`);

    setEl('v-move', `± ${fmt(movePts, 1)}`);
    setEl('v-move-pct', `± ${fmt(movePct, 3)}%  daily`);

    setEl('v-upper', `${roundStrike(upper).toLocaleString('en-IN')} (${fmt(upper, 0)})`);
    setEl('v-lower', `${roundStrike(lower).toLocaleString('en-IN')} (${fmt(lower, 0)})`);

    // Math box
    setEl('math-box',
      `<span>VIX</span> ${fmt(currentVix, 2)} &nbsp;÷&nbsp; <span>√365</span> = ${fmt(dailyVol*100, 4)}% daily vol` +
      `&nbsp;·&nbsp; <span>Spot</span> ${fmt(spot,2)} × ${fmt(dailyVol,6)} = <span>± ${fmt(movePts,1)} pts</span>`
    );

    // Timestamps → labels (last 15)
    const tail = validNifty.slice(-15);
    const labels = tail.map(x => x.t.toLocaleDateString('en-IN', { day: '2-digit', month: 'short' }));
    const closes = tail.map(x => x.c);

    buildChart(labels, closes, spot, upper, lower, currentVix, movePts);

    // Timestamp
    const now = new Date();
    setEl('ts', `UPDATED ${now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`);
    setEl('status', '✔ LIVE DATA LOADED');
    document.getElementById('status').className = '';

  } catch (err) {
    console.error(err);
    setEl('status', `✖ FETCH FAILED: ${err.message} — Yahoo Finance may block CORS in this environment.`);
    document.getElementById('status').className = 'error';
    setEl('ts', 'OFFLINE');

    // Show demo data so the UI isn't blank
    showDemo();
  }
}

function showDemo() {
  const spot = 24850.40;
  const currentVix = 13.42;
  const dailyVol = (currentVix / 100) * Math.sqrt(1/365);
  const movePts  = spot * dailyVol;
  const upper = spot + movePts; const lower = spot - movePts;

  setEl('v-spot', '24,850.40'); setEl('v-spot-chg', '▲ 112.30 (0.45%) [DEMO]');
  setEl('v-vix',  '13.42');     setEl('v-vix-chg', '▼ 0.18 pts [DEMO]');
  setEl('v-move', `± ${(movePts).toFixed(1)}`); setEl('v-move-pct', `± ${(dailyVol*100).toFixed(3)}%  daily [DEMO]`);
  setEl('v-upper', `${roundStrike(upper).toLocaleString('en-IN')} (${upper.toFixed(0)}) [DEMO]`);
  setEl('v-lower', `${roundStrike(lower).toLocaleString('en-IN')} (${lower.toFixed(0)}) [DEMO]`);
  setEl('math-box', `<span>[DEMO DATA]</span> VIX ${currentVix} ÷ √365 = ${(dailyVol*100).toFixed(4)}% daily vol · Spot ${spot} × ${dailyVol.toFixed(6)} = <span>± ${movePts.toFixed(1)} pts</span>`);

  // Simulate 15 days of closes
  const base = 24200; const closes = [];
  for (let i = 0; i < 15; i++) closes.push(base + Math.sin(i*0.6)*180 + i*30 + Math.random()*60);
  closes[14] = spot;
  const labels = Array.from({length:15}, (_,i) => {
    const d = new Date(); d.setDate(d.getDate() - (14 - i));
    return d.toLocaleDateString('en-IN', { day:'2-digit', month:'short' });
  });
  buildChart(labels, closes, spot, upper, lower, currentVix, movePts);
}

init();
</script>
</body>
</html>