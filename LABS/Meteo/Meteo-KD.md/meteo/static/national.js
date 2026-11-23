"use strict";

let nationalCatalog;
let nationalEvaluationRequest = 0;
let nationalForecastRequest = 0;
const levelSelect = document.getElementById("national-level");
const territorySelect = document.getElementById("national-territory");
const pointSelect = document.getElementById("national-point");
const nationalButton = document.getElementById("national-load");

function option(value, label) {
  const result = document.createElement("option");
  result.value = value;
  result.textContent = label;
  return result;
}

function selectedTerritory() {
  return nationalCatalog[levelSelect.value === "region" ? "regions" : "provinces"].find(item => item.id === territorySelect.value);
}

function updatePoints() {
  nationalForecastRequest++;
  setStatus("national-status", "");
  pointSelect.replaceChildren();
  for (const id of selectedTerritory().points) pointSelect.append(option(id, nationalCatalog.points[id].name));
  document.getElementById("national-forecast").hidden = true;
  loadNationalEvaluation();
}

function updateTerritories() {
  territorySelect.replaceChildren();
  const entries = nationalCatalog[levelSelect.value === "region" ? "regions" : "provinces"];
  for (const item of entries) territorySelect.append(option(item.id, item.name));
  updatePoints();
}

async function loadNationalEvaluation() {
  const request = ++nationalEvaluationRequest;
  document.getElementById("national-evaluation").hidden = true;
  setStatus("national-evaluation-status", "Caricamento valutazione…");
  try {
    const data = await getJSON(`/api/national/evaluation?level=${encodeURIComponent(levelSelect.value)}&territory=${encodeURIComponent(territorySelect.value)}`);
    if (request !== nationalEvaluationRequest) return;
    const charts = document.getElementById("national-evaluation-charts");
    charts.replaceChildren();
    for (const [target, meta] of Object.entries(targets)) {
      const result = data.territory.targets[target];
      charts.append(barChart(meta.label, meta.unit, [
        {label:"Student", value:result.student.mae},
        {label:"Teacher", value:result.teacher.mae},
        {label:"Senza distill.", value:result.supervised_student.mae},
        {label:"Persistenza", value:result.persistence.mae}
      ]));
    }
    document.getElementById("national-evaluation-context").textContent = `${data.territory.name}: ${data.territory.test_samples} esempi di test. Date delle feature dal ${dateLabel(data.split.test_first_feature_date)} ${data.split.test_first_feature_date.slice(0,4)} al ${dateLabel(data.split.test_last_feature_date)} ${data.split.test_last_feature_date.slice(0,4)}; target D+2. Peso teacher: ${number(data.distillation_weight * 100)}%. Dimensione file: student ${number(data.territory.student_size_bytes / 1024)} KiB, teacher ${number(data.teacher_size_bytes / 1024)} KiB (tre target ciascuno).`;
    document.getElementById("national-evaluation").hidden = false;
    setStatus("national-evaluation-status", "");
  } catch (error) {
    if (request === nationalEvaluationRequest) setStatus("national-evaluation-status", error.message, true);
  }
}

async function loadNationalForecast() {
  const request = ++nationalForecastRequest;
  nationalButton.disabled = true;
  document.getElementById("national-forecast").hidden = true;
  setStatus("national-status", "Calcolo previsione…");
  try {
    const query = new URLSearchParams({level:levelSelect.value, territory:territorySelect.value, point:pointSelect.value});
    const data = await getJSON(`/api/national/forecast?${query}`);
    if (request !== nationalForecastRequest) return;
    const cards = document.getElementById("national-cards");
    cards.replaceChildren();
    for (const target of Object.keys(targets)) cards.append(metricCard(target, data.student[target], [["Modello", "Student territoriale"]]));
    document.getElementById("national-date").textContent = `${data.point.name} · ${dateLabel(data.forecast_date)} ${data.forecast_date.slice(0,4)} UTC`;
    document.getElementById("national-detail").textContent = `Ultimo giorno storico: ${data.latest_history_date} UTC. ${data.disclaimer}`;
    document.getElementById("national-forecast").hidden = false;
    const tomorrow = new Date();
    tomorrow.setUTCDate(tomorrow.getUTCDate() + 1);
    setStatus("national-status", data.forecast_date !== tomorrow.toISOString().slice(0,10) ? "Attenzione: lo storico disponibile non produce una previsione per domani UTC. Aggiornare lo storico con national-refresh." : "");
  } catch (error) { if (request === nationalForecastRequest) setStatus("national-status", error.message, true); }
  finally { nationalButton.disabled = false; }
}

levelSelect.addEventListener("change", updateTerritories);
territorySelect.addEventListener("change", updatePoints);
pointSelect.addEventListener("change", () => { nationalForecastRequest++; document.getElementById("national-forecast").hidden = true; setStatus("national-status", ""); });
nationalButton.addEventListener("click", loadNationalForecast);
getJSON("/api/national/catalog").then(catalog => {
  nationalCatalog = catalog;
  updateTerritories();
  nationalButton.disabled = false;
  setStatus("national-status", "");
}).catch(error => setStatus("national-status", error.message, true));
