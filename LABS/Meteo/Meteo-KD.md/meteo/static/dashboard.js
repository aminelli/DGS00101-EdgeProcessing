"use strict";

const targets = {
  temperature_2m_max: { label: "Temperatura massima", unit: "°C", icon: "☀" },
  precipitation_sum: { label: "Precipitazioni", unit: "mm", icon: "☂" },
  wind_speed_10m_max: { label: "Vento massimo", unit: "km/h", icon: "↗" }
};
const number = value => Number.isFinite(value) ? new Intl.NumberFormat("it-IT", { maximumFractionDigits: 1 }).format(value) : "—";
const metricNumber = value => Number.isFinite(value) ? new Intl.NumberFormat("it-IT", { minimumFractionDigits: 3, maximumFractionDigits: 3 }).format(value) : "—";
const dateLabel = value => {
  const date = new Date(String(value).slice(0, 10) + "T00:00:00Z");
  return Number.isNaN(date.getTime()) ? String(value) : new Intl.DateTimeFormat("it-IT", { day: "numeric", month: "short", timeZone: "UTC" }).format(date);
};
const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};
const svgNode = (tag, attrs, text) => {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
};

async function getJSON(url) {
  const response = await fetch(url);
  let body;
  try { body = await response.json(); } catch { throw new Error(`Risposta non valida (HTTP ${response.status}).`); }
  if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `Errore HTTP ${response.status}.`);
  return body;
}

function setStatus(id, message, error = false) {
  const status = document.getElementById(id);
  status.textContent = message;
  status.classList.toggle("error", error);
}

function metricCard(target, value, details) {
  const meta = targets[target];
  const card = el("article", "metric-card");
  const top = el("div", "metric-top");
  top.append(el("span", "", meta.label), el("span", "metric-icon", meta.icon));
  const large = el("div", "metric-value", number(value) + " ");
  large.append(el("small", "", meta.unit));
  card.append(top, large);
  for (const [label, detail] of details) {
    const line = el("div", "metric-detail", label + " ");
    line.append(el("strong", "", detail));
    card.append(line);
  }
  return card;
}

function chartCard(title, unit) {
  const card = el("article", "chart-card");
  const head = el("div", "chart-head");
  head.append(el("h3", "", title), el("span", "chart-unit", unit));
  card.append(head);
  return card;
}

function lineChart(title, unit, points) {
  const card = chartCard(title, unit);
  const valid = points.filter(point => Number.isFinite(point.value));
  if (!valid.length) { card.append(el("p", "empty", "Nessun dato disponibile.")); return card; }
  const width = 360, height = 190, left = 43, right = 12, top = 14, bottom = 32;
  const values = valid.map(point => point.value);
  let min = Math.min(...values), max = Math.max(...values);
  if (min === max) { min -= 1; max += 1; }
  else { const pad = (max - min) * .1; min -= pad; max += pad; }
  const times = points.map(point => Date.parse(String(point.date).slice(0, 10) + "T00:00:00Z"));
  const span = times[times.length - 1] - times[0];
  const x = index => left + (width - left - right) * (span > 0 ? (times[index] - times[0]) / span : .5);
  const y = value => top + (height - top - bottom) * (max - value) / (max - min);
  const svg = svgNode("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": `${title}, ${unit}: andamento nel tempo` });
  for (let i = 0; i < 3; i++) {
    const v = max - (max - min) * i / 2;
    svg.append(svgNode("line", { x1: left, x2: width - right, y1: y(v), y2: y(v), class: "gridline" }),
      svgNode("text", { x: left - 6, y: y(v) + 3, "text-anchor": "end", class: "axis-label" }, number(v)));
  }
  let segment = [];
  function flush() {
    if (segment.length) svg.append(svgNode("polyline", { points: segment.join(" "), class: "chart-line" }));
    segment = [];
  }
  points.forEach((point, index) => {
    if (!Number.isFinite(point.value)) { flush(); return; }
    if (index && times[index] - times[index - 1] > 86400000) flush();
    segment.push(`${x(index)},${y(point.value)}`);
    const circle = svgNode("circle", { cx: x(index), cy: y(point.value), r: points.length <= 10 ? 3 : 1.5, class: "chart-dot" });
    circle.append(svgNode("title", {}, `${dateLabel(point.date)}: ${number(point.value)} ${unit}`));
    svg.append(circle);
  });
  flush();
  svg.append(svgNode("text", { x: left, y: height - 7, class: "axis-label" }, dateLabel(points[0].date)),
    svgNode("text", { x: width - right, y: height - 7, "text-anchor": "end", class: "axis-label" }, dateLabel(points[points.length - 1].date)));
  card.append(svg);
  return card;
}

function barChart(title, unit, entries) {
  const card = chartCard(title, unit);
  const valid = entries.filter(entry => Number.isFinite(entry.value) && entry.value >= 0);
  if (!valid.length) { card.append(el("p", "empty", "Metriche non disponibili.")); return card; }
  const width = 360, height = 40 + valid.length * 38, left = 95, right = 48;
  const max = Math.max(...valid.map(entry => entry.value), 0.001);
  const svg = svgNode("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": `${title}: confronto MAE in ${unit}` });
  valid.forEach((entry, i) => {
    const y = 16 + i * 38;
    svg.append(svgNode("text", { x: left - 8, y: y + 13, "text-anchor": "end", class: "bar-label" }, entry.label),
      svgNode("rect", { x: left, y, width: (width - left - right) * entry.value / max, height: 19, rx: 4, class: "chart-bar" }),
      svgNode("text", { x: width - right + 2, y: y + 14, class: "bar-value" }, metricNumber(entry.value)));
  });
  card.append(svg);
  return card;
}

async function loadForecast() {
  const button = document.getElementById("load-forecast");
  button.disabled = true;
  setStatus("forecast-status", "Caricamento previsione…");
  try {
    const data = await getJSON("/api/forecast");
    const cards = document.getElementById("forecast-cards");
    const charts = document.getElementById("forecast-charts");
    const table = document.getElementById("forecast-table");
    cards.replaceChildren(); charts.replaceChildren(); table.replaceChildren();
    document.getElementById("forecast-date").textContent = `Previsione per ${dateLabel(data.forecast_date)} ${String(data.forecast_date).slice(0, 4)}`;
    for (const [key, meta] of Object.entries(targets)) {
      cards.append(metricCard(key, data.machine_learning[key], [
        ["Valore", "ML"],
        ["NWP", `${number(data.nwp_next_day?.[key])} ${meta.unit}`],
        ["Ibrido", `${number(data.hybrid_next_day?.[key])} ${meta.unit}`]
      ]));
      charts.append(lineChart(meta.label, meta.unit, data.nwp.map(row => ({ date: row.date, value: row[key] }))));
    }
    for (const row of data.nwp) {
      const tr = el("tr");
      for (const text of [dateLabel(row.date), ...Object.keys(targets).map(key => `${number(row[key])} ${targets[key].unit}`)]) tr.append(el("td", "", text));
      table.append(tr);
    }
    document.getElementById("forecast-notice").textContent = `${data.hybrid_status} Peso ML: ${number(data.hybrid_ml_weight * 100)}%.`;
    document.getElementById("forecast-content").hidden = false;
    setStatus("forecast-status", "");
  } catch (error) {
    document.getElementById("forecast-content").hidden = true;
    setStatus("forecast-status", `Impossibile ottenere la previsione: ${error.message}`, true);
  } finally { button.disabled = false; }
}

let historyRequest = 0;
async function loadHistory() {
  const request = ++historyRequest;
  const days = document.getElementById("history-days").value;
  setStatus("history-status", "Caricamento dati storici…");
  try {
    const data = await getJSON(`/api/history?days=${encodeURIComponent(days)}`);
    if (request !== historyRequest) return;
    if (!data.records.length) throw new Error("Nessuna giornata completa disponibile.");
    const cards = document.getElementById("history-cards"), charts = document.getElementById("history-charts");
    cards.replaceChildren(); charts.replaceChildren();
    for (const [key, meta] of Object.entries(targets)) {
      const values = data.records.map(row => row[key]).filter(Number.isFinite);
      const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
      cards.append(metricCard(key, mean, [["Media giornaliera ·", `${values.length} giorni`]]));
      charts.append(lineChart(meta.label, meta.unit, data.records.map(row => ({ date: row.date, value: row[key] }))));
    }
    document.getElementById("history-range").textContent = `${dateLabel(data.records[0].date)} ${data.records[0].date.slice(0, 4)} — ${dateLabel(data.records.at(-1).date)} ${data.records.at(-1).date.slice(0, 4)} · ${data.records.length} giornate complete`;
    document.getElementById("history-content").hidden = false;
    setStatus("history-status", "");
  } catch (error) {
    if (request !== historyRequest) return;
    document.getElementById("history-content").hidden = true;
    setStatus("history-status", `Impossibile caricare lo storico: ${error.message}`, true);
  }
}

function addMatrix(container, rain) {
  if (!rain?.available || !Array.isArray(rain.confusion_matrix)) return;
  container.append(el("h2", "", "Classificazione della pioggia"));
  container.append(el("p", "subtitle", `Modello a due fasi · Brier score ${metricNumber(rain.brier_score)} · soglia ${number(rain.probability_threshold)}${rain.selected_for_regression ? " · selezionato per la regressione" : " · non selezionato per la regressione"}`));
  const wrap = el("div", "table-wrap matrix"), table = el("table");
  const caption = el("caption", "", "Matrice di confusione: righe reali, colonne previste");
  const head = el("thead"), hr = el("tr");
  ["", "Previsto asciutto", "Previsto piovoso"].forEach(value => hr.append(el("th", "", value)));
  head.append(hr);
  const body = el("tbody");
  rain.confusion_matrix.forEach((row, i) => {
    const tr = el("tr");
    tr.append(el("th", "", i === 0 ? "Reale asciutto" : "Reale piovoso"));
    row.forEach(value => tr.append(el("td", "", String(value))));
    body.append(tr);
  });
  table.append(caption, head, body); wrap.append(table); container.append(wrap);
}

let evaluationRequest = 0;
async function loadEvaluation() {
  const request = ++evaluationRequest;
  const kind = document.getElementById("report-type").value;
  setStatus("evaluation-status", "Caricamento risultati…");
  try {
    const data = await getJSON(`/api/evaluations/${kind}`);
    if (request !== evaluationRequest) return;
    const cards = document.getElementById("evaluation-cards"), charts = document.getElementById("evaluation-charts"), plots = document.getElementById("evaluation-plots"), matrix = document.getElementById("rain-matrix");
    cards.replaceChildren(); charts.replaceChildren(); plots.replaceChildren(); matrix.replaceChildren();
    for (const [target, meta] of Object.entries(targets)) {
      const result = data.targets[target];
      if (!result) continue;
      const model = kind === "operativo" ? result.machine_learning : result.test_selected;
      const persistence = kind === "operativo" ? result.persistence_baseline : result.test_persistence;
      if (!model || !persistence) throw new Error("Metriche di test incomplete.");
      const details = [["Indicatore", "MAE ML"], ["Persistenza", `${metricNumber(persistence.mae)} ${meta.unit}`], ["RMSE ML", `${metricNumber(model.rmse)} ${meta.unit}`], ["R² ML", metricNumber(model.r2)]];
      if (result.selected) details.unshift(["Modello", result.selected]);
      const card = metricCard(target, model.mae, details);
      card.querySelector(".metric-value").firstChild.textContent = metricNumber(model.mae) + " ";
      cards.append(card);
      const entries = [{ label: "ML", value: model.mae }, { label: "Persistenza", value: persistence.mae }];
      if (result.test_nwp) entries.push({ label: "NWP", value: result.test_nwp.mae });
      charts.append(barChart(meta.label, meta.unit, entries));
      const plot = el("article", "plot-card");
      plot.append(el("h3", "", meta.label));
      const img = document.createElement("img");
      img.src = `/api/evaluations/${kind}/plots/${target}`;
      img.alt = `Confronto storico, ML e persistenza per ${meta.label}`;
      img.addEventListener("error", () => { img.remove(); plot.append(el("p", "empty", "Grafico non disponibile. Eseguire il relativo training o esperimento.")); });
      plot.append(img); plots.append(plot);
    }
    addMatrix(matrix, data.targets.precipitation_sum?.rain_classification);
    const samples = data.split?.test_samples ?? data.sample_counts?.test;
    let context = `${kind === "operativo" ? "Modello operativo" : kind === "sperimentale" ? "Modelli sperimentali, separati dall'API" : "Esperimento avanzato NWP, separato dall'API"} · ${samples ?? "—"} esempi di test.`;
    if (data.split?.test_first_target_date) context += ` Periodo target: ${dateLabel(data.split.test_first_target_date)} ${data.split.test_first_target_date.slice(0, 4)} – ${dateLabel(data.split.test_last_target_date)} ${data.split.test_last_target_date.slice(0, 4)}.`;
    if (kind === "avanzato") context += ` ${data.nwp_availability_warning || "La disponibilità storica NWP non è certificata."}`;
    document.getElementById("evaluation-context").textContent = context;
    document.getElementById("evaluation-content").hidden = false;
    setStatus("evaluation-status", "");
  } catch (error) {
    if (request !== evaluationRequest) return;
    document.getElementById("evaluation-content").hidden = true;
    setStatus("evaluation-status", `Impossibile caricare la valutazione: ${error.message}`, true);
  }
}

const page = document.body.dataset.page;
if (page === "forecast") document.getElementById("load-forecast").addEventListener("click", loadForecast);
if (page === "history") { document.getElementById("history-days").addEventListener("change", loadHistory); loadHistory(); }
if (page === "evaluation") { document.getElementById("report-type").addEventListener("change", loadEvaluation); loadEvaluation(); }
