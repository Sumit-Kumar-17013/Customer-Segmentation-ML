/* =========================================================
   CONFIGURATION
   ---------------------------------------------------------
   Replace this with your deployed FastAPI base URL (no
   trailing slash), e.g. "https://customer-segmentation-api.onrender.com"
   ========================================================= */
const API_BASE_URL = "https://customer-segmentation-ml-68m5.onrender.com";

const ENDPOINTS = {
  predict: "/Prediction",
  health: "/Health",
};

/* =========================================================
   NOTE ON ENCODED FIELDS
   ---------------------------------------------------------
   Living_With_Encoded is documented directly in the FastAPI
   schema: 0 = Alone, 1 = Partner. The <select> option values
   below already use those encoded numbers.

   Education_Encoded is NOT documented in the FastAPI schema
   or the saved pipeline — the pipeline only knows it as a
   passthrough numeric column. The mapping below assumes the
   common 5-level ordinal encoding used in the public
   "Customer Personality Analysis" dataset this project style
   is based on (2n Cycle=0, Basic=1, Graduation=2, Master=3,
   PhD=4). If your training notebook used a different mapping,
   update the option values in index.html to match — this is
   the one field this frontend cannot verify from the files
   provided.
   ========================================================= */

/* =========================================================
   STATE
   ========================================================= */
const els = {};
let healthPollTimer = null;

/* =========================================================
   INIT
   ========================================================= */
document.addEventListener("DOMContentLoaded", () => {
  cacheEls();
  initNav();
  initClusterCanvas();
  initPipelineScrollSpy();
  initForm();
  checkApiHealth();
  healthPollTimer = setInterval(checkApiHealth, 30000);
});

function cacheEls() {
  els.nav = document.getElementById("siteNav");
  els.navBurger = document.getElementById("navBurger");
  els.mobileMenu = document.getElementById("mobileMenu");
  els.apiStatusDot = document.getElementById("apiStatusDot");
  els.apiStatusLabel = document.getElementById("apiStatusLabel");
  els.toastStack = document.getElementById("toastStack");

  els.form = document.getElementById("predictForm");
  els.predictBtn = document.getElementById("predictBtn");
  els.resetBtn = document.getElementById("resetBtn");
  els.formNote = document.getElementById("formNote");

  els.resultPlaceholder = document.getElementById("resultPlaceholder");
  els.resultScanning = document.getElementById("resultScanning");
  els.resultCard = document.getElementById("resultCard");
  els.resultError = document.getElementById("resultError");
  els.resultErrorMessage = document.getElementById("resultErrorMessage");
  els.resultCluster = document.getElementById("resultCluster");
  els.resultSegment = document.getElementById("resultSegment");
  els.resultMessage = document.getElementById("resultMessage");
  els.analyzeAnotherBtn = document.getElementById("analyzeAnotherBtn");

  els.pipelineNodes = Array.from(document.querySelectorAll(".pipeline-node"));
}

/* =========================================================
   NAVIGATION (mobile menu + scroll shadow)
   ========================================================= */
function initNav() {
  els.navBurger.addEventListener("click", () => {
    const isOpen = els.mobileMenu.classList.toggle("is-open");
    els.navBurger.setAttribute("aria-expanded", String(isOpen));
  });

  els.mobileMenu.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
      els.mobileMenu.classList.remove("is-open");
      els.navBurger.setAttribute("aria-expanded", "false");
    });
  });
}

/* =========================================================
   HERO CANVAS — animated K-Means style cluster formation
   ========================================================= */
function initClusterCanvas() {
  const canvas = document.getElementById("clusterCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  let width, height, dpr;
  const POINT_COUNT = 70;
  let points = [];
  let centroids = [];
  let phase = 0; // 0 = scattered, 1 = converging, 2 = held
  let phaseTimer = 0;

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = canvas.clientWidth = canvas.parentElement.offsetWidth;
    height = canvas.clientHeight = canvas.parentElement.offsetHeight;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    setup();
  }

  function setup() {
    centroids = [
      { x: width * 0.32, y: height * 0.42, color: "56, 189, 248" },
      { x: width * 0.7, y: height * 0.6, color: "251, 191, 36" },
    ];
    points = Array.from({ length: POINT_COUNT }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      scatterX: Math.random() * width,
      scatterY: Math.random() * height,
      cluster: Math.random() < 0.5 ? 0 : 1,
      r: 1.6 + Math.random() * 1.8,
    }));
  }

  function targetFor(p) {
    const c = centroids[p.cluster];
    const angle = (p.x + p.y) * 0.01;
    const spread = 70;
    return {
      x: c.x + Math.cos(angle + p.cluster * 3) * spread * Math.random(),
      y: c.y + Math.sin(angle) * spread * Math.random(),
    };
  }

  function draw() {
    ctx.clearRect(0, 0, width, height);

    // connecting lines (subtle)
    ctx.lineWidth = 1;
    for (let i = 0; i < points.length; i++) {
      for (let j = i + 1; j < points.length; j++) {
        const a = points[i], b = points[j];
        if (a.cluster !== b.cluster) continue;
        const dx = a.x - b.x, dy = a.y - b.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 60) {
          ctx.strokeStyle = `rgba(${centroids[a.cluster].color}, ${0.12 * (1 - dist / 60)})`;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }

    // points
    points.forEach((p) => {
      ctx.beginPath();
      ctx.fillStyle = `rgba(${centroids[p.cluster].color}, 0.85)`;
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
    });

    // centroids (visible once converged)
    if (phase >= 1) {
      centroids.forEach((c) => {
        ctx.beginPath();
        ctx.strokeStyle = `rgba(${c.color}, 0.5)`;
        ctx.lineWidth = 1.5;
        ctx.arc(c.x, c.y, 84, 0, Math.PI * 2);
        ctx.stroke();
      });
    }
  }

  function step() {
    phaseTimer++;

    if (phase === 0 && phaseTimer > 60) {
      phase = 1;
      phaseTimer = 0;
      points.forEach((p) => { p.target = targetFor(p); });
    } else if (phase === 1 && phaseTimer > 140) {
      phase = 2;
      phaseTimer = 0;
    } else if (phase === 2 && phaseTimer > 220) {
      phase = 0;
      phaseTimer = 0;
      points.forEach((p) => {
        p.target = { x: p.scatterX, y: p.scatterY };
      });
    }

    const ease = phase === 0 && phaseTimer === 0 ? 1 : 0.035;
    points.forEach((p) => {
      const t = p.target || { x: p.scatterX, y: p.scatterY };
      p.x += (t.x - p.x) * ease;
      p.y += (t.y - p.y) * ease;
    });

    draw();
    if (!prefersReducedMotion) requestAnimationFrame(step);
  }

  window.addEventListener("resize", debounce(resize, 200));
  resize();
  draw();
  if (!prefersReducedMotion) requestAnimationFrame(step);
}

/* =========================================================
   PIPELINE — highlight nodes as they scroll into view
   ========================================================= */
function initPipelineScrollSpy() {
  if (!("IntersectionObserver" in window) || els.pipelineNodes.length === 0) return;
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const idx = Number(entry.target.dataset.node);
          setTimeout(() => entry.target.classList.add("is-active"), idx * 90);
        }
      });
    },
    { threshold: 0.5 }
  );
  els.pipelineNodes.forEach((node) => observer.observe(node));
}

/* =========================================================
   API HEALTH CHECK
   ========================================================= */
async function checkApiHealth() {
  if (!API_BASE_URL || API_BASE_URL.includes("YOUR_RENDER_FASTAPI_URL")) {
    setApiStatus("offline", "API URL not set");
    return;
  }
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    const res = await fetch(`${API_BASE_URL}${ENDPOINTS.health}`, {
      method: "GET",
      signal: controller.signal,
    });
    clearTimeout(timeout);
    if (res.ok) {
      setApiStatus("online", "API online");
    } else {
      setApiStatus("offline", "API offline");
    }
  } catch (err) {
    console.error("Health check failed:", err);
    setApiStatus("offline", "API offline");
  }
}

function setApiStatus(state, label) {
  els.apiStatusDot.classList.remove("is-online", "is-offline");
  els.apiStatusDot.classList.add(state === "online" ? "is-online" : "is-offline");
  els.apiStatusLabel.textContent = label;
}

/* =========================================================
   FORM — validation, submit, reset
   ========================================================= */
const FIELD_IDS = [
  "Age", "Income", "Recency", "Customer_Tenure_Days", "total_spend",
  "total_purchase", "total_campaigns", "children", "family_size",
  "Education_Encoded", "Living_With_Encoded", "NumWebVisitsMonth",
];

function initForm() {
  els.form.addEventListener("submit", handleSubmit);
  els.resetBtn.addEventListener("click", handleReset);
  els.analyzeAnotherBtn.addEventListener("click", handleReset);

  FIELD_IDS.forEach((id) => {
    const input = document.getElementById(id);
    input.addEventListener("input", () => clearFieldError(id));
  });
}

function validateForm() {
  let firstInvalid = null;
  const values = {};

  FIELD_IDS.forEach((id) => {
    const input = document.getElementById(id);
    const raw = input.value;
    const isSelect = input.tagName === "SELECT";
    let valid = raw !== "" && raw !== null;

    if (valid && !isSelect) {
      const num = Number(raw);
      valid = !Number.isNaN(num);
      if (valid && input.min !== "" && num < Number(input.min)) valid = false;
      if (valid && input.max !== "" && num > Number(input.max)) valid = false;
      if (valid) values[id] = num;
    } else if (valid && isSelect) {
      values[id] = Number(raw);
    }

    if (!valid) {
      setFieldError(id, "Please enter a valid value.");
      if (!firstInvalid) firstInvalid = input;
    } else {
      clearFieldError(id);
    }
  });

  if (firstInvalid) {
    firstInvalid.focus();
    return null;
  }
  return values;
}

function setFieldError(id, message) {
  const input = document.getElementById(id);
  const field = input.closest(".field");
  const errEl = document.getElementById(`err-${id}`);
  field.classList.add("has-error");
  if (errEl) errEl.textContent = message;
  // restart shake animation
  field.style.animation = "none";
  requestAnimationFrame(() => { field.style.animation = ""; });
}

function clearFieldError(id) {
  const input = document.getElementById(id);
  const field = input.closest(".field");
  const errEl = document.getElementById(`err-${id}`);
  field.classList.remove("has-error");
  if (errEl) errEl.textContent = "";
}

async function handleSubmit(e) {
  e.preventDefault();
  els.formNote.textContent = "";

  const values = validateForm();
  if (!values) {
    els.formNote.textContent = "Please fix the highlighted fields before continuing.";
    return;
  }

  if (!API_BASE_URL || API_BASE_URL.includes("YOUR_RENDER_FASTAPI_URL")) {
    showToast("Set API_BASE_URL in script.js before predicting.", "error");
    els.formNote.textContent = "API_BASE_URL is not configured yet.";
    return;
  }

  setLoadingState(true);

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 20000);

    const response = await fetch(`${API_BASE_URL}${ENDPOINTS.predict}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (!response.ok) {
      await handleHttpError(response);
      return;
    }

    const data = await response.json();
    if (
      data == null ||
      typeof data.cluster === "undefined" ||
      typeof data.segment === "undefined" ||
      typeof data.message === "undefined"
    ) {
      throw new Error("Unexpected response shape from API.");
    }

    renderResult(data);
  } catch (err) {
    console.error("Prediction request failed:", err);
    if (err.name === "AbortError") {
      showResultError("The request timed out. Check that your API is awake and reachable, then try again.");
    } else {
      showResultError("Couldn't reach the prediction API. Check your connection or try again shortly.");
    }
    showToast("Prediction request failed.", "error");
  } finally {
    setLoadingState(false);
  }
}

async function handleHttpError(response) {
  let detail = "";
  try {
    const body = await response.json();
    detail = body?.detail ? (Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join(" ") : body.detail) : "";
  } catch (_) {
    /* response wasn't JSON — ignore */
  }

  let message;
  switch (response.status) {
    case 400:
      message = "The API rejected the request. Please check the values you entered.";
      break;
    case 401:
      message = "The API requires authentication that this frontend isn't sending yet.";
      break;
    case 422:
      message = detail || "One or more fields don't match what the API expects.";
      break;
    case 500:
      message = "The API had a problem generating a prediction. Please try again.";
      break;
    default:
      message = `The API returned an unexpected error (${response.status}).`;
  }

  console.error("API error response:", response.status, detail);
  showResultError(message);
  showToast(message, "error");
}

function setLoadingState(isLoading) {
  els.predictBtn.classList.toggle("is-loading", isLoading);
  els.predictBtn.disabled = isLoading;
  els.resetBtn.disabled = isLoading;

  if (isLoading) {
    show(els.resultScanning);
    hide(els.resultPlaceholder);
    hide(els.resultCard);
    hide(els.resultError);
  } else {
    hide(els.resultScanning);
  }
}

function renderResult(data) {
  const cluster = data.cluster;
  els.resultCluster.textContent = cluster;
  els.resultSegment.textContent = data.segment;
  els.resultMessage.textContent = data.message;
  els.resultCard.dataset.cluster = String(cluster);

  hide(els.resultPlaceholder);
  hide(els.resultError);
  hide(els.resultScanning);
  show(els.resultCard);

  showToast("Segment predicted successfully.", "success");
}

function showResultError(message) {
  els.resultErrorMessage.textContent = message;
  hide(els.resultPlaceholder);
  hide(els.resultCard);
  hide(els.resultScanning);
  show(els.resultError);
}

function handleReset() {
  els.form.reset();
  FIELD_IDS.forEach(clearFieldError);
  els.formNote.textContent = "";
  hide(els.resultCard);
  hide(els.resultError);
  hide(els.resultScanning);
  show(els.resultPlaceholder);
}

/* =========================================================
   TOASTS
   ========================================================= */
function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast${type === "error" ? " is-error" : type === "success" ? " is-success" : ""}`;
  toast.textContent = message;
  els.toastStack.appendChild(toast);

  setTimeout(() => {
    toast.classList.add("is-leaving");
    setTimeout(() => toast.remove(), 240);
  }, 4200);
}

/* =========================================================
   HELPERS
   ========================================================= */
function show(el) { if (el) el.hidden = false; }
function hide(el) { if (el) el.hidden = true; }

function debounce(fn, wait) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), wait);
  };
}
