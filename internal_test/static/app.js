const elements = {
  benchmarkName: document.querySelector("#benchmarkName"),
  clock: document.querySelector("#clock"),
  controllerStatus: document.querySelector("#controllerStatus"),
  emptyChart: document.querySelector("#emptyChart"),
  endpointType: document.querySelector("#endpointType"),
  jobCount: document.querySelector("#jobCount"),
  jobFilter: document.querySelector("#jobFilter"),
  jobList: document.querySelector("#jobList"),
  jobRequests: document.querySelector("#jobRequests"),
  jobState: document.querySelector("#jobState"),
  jobTitle: document.querySelector("#jobTitle"),
  model: document.querySelector("#model"),
  observedHelp: document.querySelector("#observedHelp"),
  productionHelp: document.querySelector("#productionHelp"),
  productionRate: document.querySelector("#productionRate"),
  reportingJobs: document.querySelector("#reportingJobs"),
  schedule: document.querySelector("#schedule"),
  smoothness: document.querySelector("#smoothness"),
  startRun: document.querySelector("#startRun"),
  statusDot: document.querySelector("#statusDot"),
  targetHelp: document.querySelector("#targetHelp"),
  targetLabel: document.querySelector("#targetLabel"),
  testRate: document.querySelector("#testRate"),
  totalRequests: document.querySelector("#totalRequests"),
  trafficChart: document.querySelector("#trafficChart"),
  trafficShape: document.querySelector("#trafficShape"),
};

const chartPadding = { top: 30, right: 18, bottom: 34, left: 52 };
let snapshot = null;
let trafficData = null;
let selectedJobId = null;
let selectedSecond = 0;
let selectionPinned = false;

elements.startRun.addEventListener("click", async () => {
  elements.startRun.disabled = true;
  elements.startRun.textContent = "Starting…";
  try {
    const response = await fetch("/api/run/start", { method: "POST" });
    if (!response.ok) {
      const result = await response.json();
      throw new Error(result.error || `start returned ${response.status}`);
    }
    await refresh();
  } catch (error) {
    elements.startRun.textContent = "Start failed";
    elements.controllerStatus.textContent = error.message;
    elements.statusDot.classList.add("failed");
  }
});
elements.jobFilter.addEventListener("input", renderJobList);
elements.trafficChart.addEventListener("pointermove", scrubChart);
elements.trafficChart.addEventListener("pointerdown", scrubChart);
function scrubChart(event) {
  if (!snapshot) return;
  const bounds = elements.trafficChart.getBoundingClientRect();
  const plotWidth = bounds.width - chartPadding.left - chartPadding.right;
  const pointerX = event.clientX - bounds.left - chartPadding.left;
  const fraction = Math.min(1, Math.max(0, pointerX / plotWidth));
  const elapsedSeconds = Math.max(
    0,
    (trafficData?.second_buckets.length || 1) - 1,
  );
  const timelineSeconds = Math.floor(snapshot.plan.timeline_minutes * 60);
  selectionPinned = true;
  selectedSecond = Math.min(
    elapsedSeconds,
    Math.round(fraction * timelineSeconds),
  );
  drawChart();
}
window.addEventListener("resize", drawChart);

async function refresh() {
  try {
    const response = await fetch("/api/snapshot", { cache: "no-store" });
    if (!response.ok) throw new Error(`snapshot returned ${response.status}`);
    snapshot = await response.json();
    if (!selectedJobId && snapshot.plan.jobs.length) {
      selectedJobId = snapshot.plan.jobs[0].job_id;
    }
    await refreshTraffic();
    const elapsedSeconds = Math.floor(snapshot.virtual_minute * 60);
    if (!selectionPinned || selectedSecond > elapsedSeconds) {
      selectedSecond = elapsedSeconds;
    }
    render();
  } catch (error) {
    elements.controllerStatus.textContent = "Disconnected";
    elements.statusDot.classList.add("failed");
  }
}

async function refreshTraffic() {
  if (!selectedJobId) return;
  const response = await fetch(
    `/api/jobs/${encodeURIComponent(selectedJobId)}/traffic`,
    { cache: "no-store" },
  );
  if (!response.ok) throw new Error(`traffic returned ${response.status}`);
  trafficData = await response.json();
}

function render() {
  const jobs = snapshot.plan.jobs;
  const total = Object.values(snapshot.jobs)
    .reduce((sum, job) => sum + job.total_requests, 0);
  const reporting = Object.values(snapshot.jobs)
    .filter((job) => job.total_requests > 0).length;
  const currentMinute = Math.min(snapshot.virtual_minute, snapshot.plan.timeline_minutes);

  elements.benchmarkName.textContent = snapshot.plan.benchmark_id;
  const drainMinutes = snapshot.plan.timeline_minutes - snapshot.plan.horizon_minutes;
  elements.schedule.textContent = `${formatNumber(snapshot.plan.horizon_minutes)} min + ${formatNumber(drainMinutes)} min drain`;
  elements.totalRequests.textContent = total.toLocaleString();
  elements.reportingJobs.textContent = `${reporting} / ${jobs.length}`;
  elements.jobCount.textContent = jobs.length;
  elements.clock.textContent = `T+${currentMinute.toFixed(1)} min`;
  elements.controllerStatus.textContent = snapshot.controller.status;
  elements.statusDot.classList.toggle("failed", snapshot.controller.status === "failed");
  elements.statusDot.classList.toggle("idle", snapshot.controller.status === "idle");
  elements.startRun.disabled = snapshot.controller.status !== "idle";
  elements.startRun.textContent = snapshot.controller.status === "idle"
    ? "Start 3-hour run"
    : snapshot.controller.status === "completed"
      ? "Run complete"
      : snapshot.controller.status === "failed"
        ? "Run failed"
        : "Run in progress";
  renderJobList();
  renderJobDetail();
}

function renderJobList() {
  if (!snapshot) return;
  const query = elements.jobFilter.value.trim().toLowerCase();
  const jobs = snapshot.plan.jobs.filter((job) => (
    job.display_name.toLowerCase().includes(query)
    || job.workload_type.toLowerCase().includes(query)
    || job.operation.toLowerCase().includes(query)
    || String(job.model).toLowerCase().includes(query)
    || String(job.traffic_pattern).toLowerCase().includes(query)
  ));
  elements.jobList.replaceChildren(...jobs.map((job) => {
    const button = document.createElement("button");
    const data = snapshot.jobs[job.job_id];
    button.type = "button";
    button.className = `job-button${job.job_id === selectedJobId ? " selected" : ""}`;
    button.innerHTML = `
      <span>${escapeHtml(job.display_name)}</span>
      <small>
        <span>${escapeHtml(job.workload_type)} · ${escapeHtml(job.kind)} · ${escapeHtml(job.traffic_pattern)}</span>
        <b>${data.total_requests.toLocaleString()}</b>
      </small>
    `;
    button.addEventListener("click", async () => {
      selectedJobId = job.job_id;
      trafficData = null;
      selectionPinned = false;
      renderJobList();
      try {
        await refreshTraffic();
        selectedSecond = Math.floor(snapshot.virtual_minute * 60);
        renderJobDetail();
      } catch (error) {
        elements.controllerStatus.textContent = error.message;
      }
    });
    return button;
  }));
}

function renderJobDetail() {
  const job = selectedPlan();
  if (!job) return;
  const data = snapshot.jobs[job.job_id];
  const state = jobState(job, snapshot.virtual_minute, snapshot.controller.status);

  elements.endpointType.textContent = `${job.workload_type} · ${job.operation} · ${job.endpoint_type}`;
  elements.jobTitle.textContent = job.display_name;
  elements.model.textContent = job.model;
  elements.jobState.textContent = state;
  elements.jobState.classList.toggle("active", state === "active");
  elements.trafficShape.textContent = job.traffic_pattern;
  elements.smoothness.textContent = job.smoothness == null
    ? (job.kind === "batch"
      ? "Finite dataset released at start"
      : job.rate_ramp_minutes == null
        ? "Configured arrival process"
        : `Ramp over ${formatNumber(job.rate_ramp_minutes)} min`)
    : `Smoothness ${formatNumber(job.smoothness)}`;
  if (job.kind === "batch") {
    const progress = job.record_count
      ? Math.min(100, data.total_requests / job.record_count * 100)
      : 0;
    elements.productionRate.textContent = `${formatNumber(job.record_count)} requests`;
    elements.productionHelp.textContent = "Full finite dataset";
    elements.targetLabel.textContent = "Batch progress";
    elements.testRate.textContent = `${progress.toFixed(2)}%`;
    elements.targetHelp.textContent = `${data.total_requests.toLocaleString()} received`;
  } else {
    elements.productionRate.textContent = `${formatNumber(job.production_rate)} RPS`;
    elements.productionHelp.textContent = "Uncapped benchmark rate";
    elements.targetLabel.textContent = "Expected rate";
    elements.testRate.textContent = `${formatNumber(job.production_rate)} RPS`;
    elements.targetHelp.textContent = "Full configured workload";
  }
  elements.jobRequests.textContent = data.total_requests.toLocaleString();
  const currentRps = trafficData?.job_id === job.job_id
    ? trafficData.second_buckets.at(-1) || 0
    : 0;
  elements.observedHelp.textContent = `${currentRps.toLocaleString()} req/s now`;
  elements.emptyChart.style.display = data.total_requests ? "none" : "grid";

  drawChart();
}

function drawChart() {
  const job = selectedPlan();
  if (!job || !snapshot) return;
  const canvas = elements.trafficChart;
  const bounds = canvas.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(bounds.width * ratio);
  canvas.height = Math.round(bounds.height * ratio);
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);

  const width = bounds.width;
  const height = bounds.height;
  const padding = chartPadding;
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const horizonMinutes = Number(snapshot.plan.timeline_minutes);
  const horizonSeconds = Math.floor(horizonMinutes * 60);
  const observed = trafficData?.job_id === job.job_id
    ? trafficData.second_buckets
    : [0];
  const rateMaximum = Math.max(1, ...observed) * 1.12;

  context.clearRect(0, 0, width, height);
  context.font = '10px Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
  context.fillStyle = "#8792a3";
  context.strokeStyle = "#edf1f6";
  context.lineWidth = 1;
  for (let index = 0; index <= 4; index += 1) {
    const y = padding.top + (plotHeight * index / 4);
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(width - padding.right, y);
    context.stroke();
    const rateValue = rateMaximum * (1 - index / 4);
    context.fillText(rateValue.toFixed(rateValue < 10 ? 1 : 0), 5, y + 3);
  }
  context.fillText("req/s", 8, 14);
  [0, 30, 60, 90, 120, 150, 180, 210]
    .filter((minute) => minute <= horizonMinutes)
    .forEach((minute) => {
      const x = padding.left + plotWidth * minute / horizonMinutes;
      context.fillText(`${minute} min`, x - 16, height - 7);
    });

  plotLine(context, observed, horizonSeconds, rateMaximum, padding, plotWidth, plotHeight, "#5c9fe4", 2);

  const cursorX = padding.left + plotWidth * selectedSecond / horizonSeconds;
  context.strokeStyle = "#cfd7e2";
  context.lineWidth = 1;
  context.setLineDash([3, 4]);
  context.beginPath();
  context.moveTo(cursorX, padding.top);
  context.lineTo(cursorX, padding.top + plotHeight);
  context.stroke();
  context.setLineDash([]);

  const selectedRps = observed[selectedSecond] || 0;
  const pointY = padding.top + plotHeight * (1 - selectedRps / rateMaximum);
  context.fillStyle = "#5c9fe4";
  context.beginPath();
  context.arc(cursorX, pointY, 4, 0, Math.PI * 2);
  context.fill();

  const readout = `${(selectedSecond / 60).toFixed(2)} min · ${selectedRps.toLocaleString()} req/s`;
  const readoutWidth = context.measureText(readout).width + 16;
  const readoutX = Math.min(
    width - padding.right - readoutWidth,
    Math.max(padding.left, cursorX - readoutWidth / 2),
  );
  const readoutY = Math.max(padding.top + 5, pointY - 34);
  context.save();
  context.shadowColor = "rgb(40 57 78 / 12%)";
  context.shadowBlur = 10;
  context.shadowOffsetY = 3;
  context.fillStyle = "#ffffff";
  context.fillRect(readoutX, readoutY, readoutWidth, 24);
  context.restore();
  context.strokeStyle = "#e3e8ef";
  context.strokeRect(readoutX, readoutY, readoutWidth, 24);
  context.fillStyle = "#364152";
  context.fillText(readout, readoutX + 8, readoutY + 16);
}

function plotLine(context, values, horizon, maximum, padding, plotWidth, plotHeight, color, lineWidth) {
  context.strokeStyle = color;
  context.lineWidth = lineWidth;
  context.lineJoin = "round";
  context.beginPath();
  values.forEach((value, second) => {
    const x = padding.left + plotWidth * second / horizon;
    const y = padding.top + plotHeight * (1 - value / maximum);
    if (second === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.stroke();
}

function jobState(job, minute, controllerStatus) {
  if (controllerStatus === "idle") return "idle";
  if (controllerStatus === "starting") return "pending";
  if (controllerStatus === "failed") return "failed";
  if (minute < job.start_minute) return "pending";
  if (job.kind === "batch") {
    const received = snapshot.jobs[job.job_id].total_requests;
    if (job.record_count && received >= job.record_count) return "complete";
    if (controllerStatus === "running") return "active";
    return "incomplete";
  }
  if (minute < job.stop_minute && controllerStatus === "running") return "active";
  return "complete";
}

function selectedPlan() {
  return snapshot?.plan.jobs.find((job) => job.job_id === selectedJobId);
}

function formatNumber(value) {
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

refresh();
setInterval(refresh, 1000);
