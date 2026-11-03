const closes = Array.from({ length: 120 }, (_, index) => {
  const trend = index * 0.075;
  const cycle = Math.sin(index / 7) * 1.8 + Math.sin(index / 17) * 1.1;
  const shock = index === 44 ? -3.2 : index === 45 ? -1.7 : index === 83 ? 2.4 : 0;
  return 100 + trend + cycle + shock;
});
const $ = (selector) => document.querySelector(selector);
function movingAverage(values, window) { return values.slice(-window).reduce((sum, value) => sum + value, 0) / window; }
function run() {
  const short = Number($("#short").value), long = Number($("#long").value), cost = Number($("#cost").value) / 10000;
  const strategy = [10000], benchmark = [10000], signals = []; let position = 0, trades = 0;
  closes.forEach((close, index) => {
    const signal = index >= long - 1 && movingAverage(closes.slice(0, index + 1), short) > movingAverage(closes.slice(0, index + 1), long) ? 1 : 0;
    if (signal !== position) { trades++; position = signal; }
    signals.push(signal);
    const previousClose = closes[index - 1] || closes[0];
    const change = close / previousClose;
    strategy.push(strategy.at(-1) * (signal ? change * (1 - cost) : 1));
    benchmark.push(benchmark.at(-1) * change);
  });
  if (position) { strategy[strategy.length - 1] *= 1 - cost; trades++; }
  const values = strategy.slice(1), returns = values.slice(1).map((value, index) => value / values[index] - 1);
  let peak = values[0], drawdown = 0;
  values.forEach((value) => { peak = Math.max(peak, value); drawdown = Math.max(drawdown, (peak - value) / peak); });
  const mean = returns.reduce((a, b) => a + b, 0) / Math.max(returns.length, 1);
  const volatility = Math.sqrt(returns.reduce((sum, value) => sum + (value - mean) ** 2, 0) / Math.max(returns.length, 1)) * Math.sqrt(252);
  $("#short-output").textContent = short; $("#long-output").textContent = long; $("#cost-output").textContent = `${(cost * 100).toFixed(2)}%`;
  $("#return").textContent = `${((values.at(-1) / 10000 - 1) * 100).toFixed(2)}%`; $("#drawdown").textContent = `${(drawdown * 100).toFixed(2)}%`; $("#volatility").textContent = `${(volatility * 100).toFixed(2)}%`; $("#trades").textContent = trades;
  $("#position").textContent = signals.at(-1) ? "Long" : "Flat"; $("#date").textContent = `Session ${signals.length}`;
  drawChart(strategy, benchmark); drawHero(strategy); $("#hero-return").textContent = `${((values.at(-1) / 10000 - 1) * 100).toFixed(2)}%`;
}
function points(values, width, height, min, max) { return values.map((value, index) => `${(index / (values.length - 1)) * width},${height - ((value - min) / (max - min)) * height}`).join(" "); }
function drawChart(strategy, benchmark) {
  const svg = $("#chart"), width = svg.clientWidth || 800, height = 300, all = [...strategy, ...benchmark], min = Math.min(...all) * .995, max = Math.max(...all) * 1.005;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`); const grid = [0.2, 0.5, 0.8].map((ratio) => `<line class="gridline" x1="0" y1="${height * ratio}" x2="${width}" y2="${height * ratio}"></line>`).join("");
  svg.innerHTML = `${grid}<polyline class="benchmark-line" points="${points(benchmark, width, height, min, max)}"></polyline><polyline class="strategy-line" points="${points(strategy, width, height, min, max)}"></polyline><text class="axis-label" x="4" y="13">$${Math.round(max).toLocaleString()}</text><text class="axis-label" x="4" y="${height - 4}">$${Math.round(min).toLocaleString()}</text>`;
}
function drawHero(strategy) { const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg"), width = 400, height = 220; svg.setAttribute("viewBox", `0 0 ${width} ${height}`); svg.innerHTML = `<polyline points="${points(strategy, width, height, Math.min(...strategy), Math.max(...strategy))}" fill="none" stroke="#1c72e8" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"></polyline>`; $("#hero-chart").replaceChildren(svg); }
["short", "long", "cost"].forEach((id) => $(`#${id}`).oninput = run); $("#run").onclick = run; window.addEventListener("resize", run); run();
