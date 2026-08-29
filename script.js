/**
 * ==============================================================================
 * CustomerIQ - Machine Learning Customer Segmentation Frontend Controller
 * Model: Preprocessing -> log1p(total_spend) -> StandardScaler -> PCA -> K-Means
 * Backend: FastAPI (Deployed on Render / Localhost)
 * ==============================================================================
 */

// ==============================================================================
// 1. FASTAPI BACKEND CONFIGURATION
// ==============================================================================
/**
 * REPLACE "YOUR_RENDER_FASTAPI_URL" with your live Render backend URL,
 * for example: "https://customer-segmentation-api.onrender.com"
 * 
 * Note: If you leave this as "YOUR_RENDER_FASTAPI_URL" or test locally,
 * you can also click the API STATUS badge in the top bar to test/change
 * your URL in real-time without re-deploying!
 */
const API_BASE_URL = "https://customer-segmentation-ml-68m5.onrender.com";

// Local storage key for runtime URL override (useful during testing)
const STORAGE_KEY_API_URL = "customeriq_api_base_url";

// Get active API URL (checks localStorage first, then falls back to API_BASE_URL or window.location)
function getActiveApiUrl() {
  const savedUrl = localStorage.getItem(STORAGE_KEY_API_URL);
  if (savedUrl && savedUrl.trim() !== "") {
    return savedUrl.trim().replace(/\/+$/, "");
  }
  if (API_BASE_URL && API_BASE_URL !== "YOUR_RENDER_FASTAPI_URL") {
    return API_BASE_URL.replace(/\/+$/, "");
  }
  // Default fallback for local testing
  return "http://localhost:8000";
}

// Global state
const appState = {
  isPredicting: false,
  apiStatus: "checking", // 'online' | 'offline' | 'checking'
  lastLatencyMs: null,
  lastPrediction: null
};

// ==============================================================================
// 2. DOM ELEMENTS
// ==============================================================================
const DOM = {
  // Form Elements
  form: document.getElementById("customer-segmentation-form"),
  btnPredict: document.getElementById("btn-predict"),
  btnReset: document.getElementById("btn-reset"),
  
  // Fields (Exact match to FastAPI ModelData)
  fieldAge: document.getElementById("field-age"),
  fieldIncome: document.getElementById("field-income"),
  fieldEducation: document.getElementById("field-education"),
  fieldLivingWith: document.getElementById("field-living-with"),
  fieldChildren: document.getElementById("field-children"),
  fieldFamilySize: document.getElementById("field-family-size"),
  fieldRecency: document.getElementById("field-recency"),
  fieldTenure: document.getElementById("field-tenure"),
  fieldWebVisits: document.getElementById("field-web-visits"),
  fieldSpend: document.getElementById("field-spend"),
  fieldPurchases: document.getElementById("field-purchases"),
  fieldCampaigns: document.getElementById("field-campaigns"),

  // Result Section
  placeholderCard: document.getElementById("result-placeholder-card"),
  resultCard: document.getElementById("prediction-result-card"),
  clusterBadge: document.getElementById("result-cluster-badge"),
  segmentHeading: document.getElementById("result-segment-name"),
  insightText: document.getElementById("result-insight-text"),
  recommendationBullets: document.getElementById("result-action-bullets"),
  resultTimestamp: document.getElementById("result-timestamp"),
  snapshotSpend: document.getElementById("snapshot-spend"),
  snapshotIncome: document.getElementById("snapshot-income"),
  snapshotPurchases: document.getElementById("snapshot-purchases"),
  rawPayloadBtn: document.getElementById("raw-payload-btn"),
  rawPayloadView: document.getElementById("raw-payload-view"),

  // API Status & Modal
  apiStatusBadge: document.getElementById("api-status-badge"),
  statusDot: document.getElementById("status-dot"),
  statusText: document.getElementById("status-text"),
  configModal: document.getElementById("api-config-modal"),
  modalCloseBtn: document.getElementById("modal-close-btn"),
  apiUrlInput: document.getElementById("api-url-input"),
  btnSaveApiUrl: document.getElementById("btn-save-api-url"),
  btnPingHealth: document.getElementById("btn-ping-health"),
  healthTestResult: document.getElementById("health-test-result"),

  // Presets
  presetHighValue: document.getElementById("preset-high-value"),
  presetBrowsing: document.getElementById("preset-browsing"),
  presetBalanced: document.getElementById("preset-balanced"),

  // Mobile menu
  mobileMenuBtn: document.getElementById("mobile-menu-btn"),
  mobileNavDrawer: document.getElementById("mobile-nav-drawer"),

  // Toast Container
  toastContainer: document.getElementById("toast-container")
};

// ==============================================================================
// 3. INITIALIZATION & EVENT LISTENERS
// ==============================================================================
document.addEventListener("DOMContentLoaded", () => {
  initApiStatus();
  initFormEventListeners();
  initPresets();
  initModalListeners();
  initMobileMenu();
  renderHeroClusterNodes();

  // Load initial preset (High-Value Archetype) as default sample values
  loadPresetData("high-value");
});

// ==============================================================================
// 4. API HEALTH MONITORING & CONFIGURATION
// ==============================================================================
async function checkApiHealth(customUrl = null) {
  const targetUrl = customUrl || getActiveApiUrl();
  updateStatusUI("checking", "Pinging API...");

  const startTime = performance.now();
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 6000); // 6s timeout

  try {
    const response = await fetch(`${targetUrl}/Health`, {
      method: "GET",
      signal: controller.signal,
      headers: { "Accept": "application/json" }
    });

    clearTimeout(timeoutId);
    const latency = Math.round(performance.now() - startTime);
    appState.lastLatencyMs = latency;

    if (response.ok) {
      const data = await response.json();
      appState.apiStatus = "online";
      updateStatusUI("online", `API Online (${latency}ms)`);
      return { ok: true, data, latency };
    } else {
      appState.apiStatus = "offline";
      updateStatusUI("offline", `API Error (${response.status})`);
      return { ok: false, status: response.status };
    }
  } catch (err) {
    clearTimeout(timeoutId);
    appState.apiStatus = "offline";
    const errorMsg = err.name === "AbortError" ? "Timeout" : "Offline";
    updateStatusUI("offline", `API ${errorMsg}`);
    return { ok: false, error: err.message };
  }
}

function updateStatusUI(status, text) {
  if (!DOM.statusDot || !DOM.statusText) return;
  
  DOM.statusDot.className = `status-dot ${status}`;
  DOM.statusText.textContent = text;
}

function initApiStatus() {
  if (DOM.apiStatusBadge) {
    DOM.apiStatusBadge.addEventListener("click", () => {
      openConfigModal();
    });
  }
  // Check health on initial boot
  checkApiHealth();
}

function openConfigModal() {
  if (!DOM.configModal) return;
  DOM.apiUrlInput.value = getActiveApiUrl();
  DOM.healthTestResult.style.display = "none";
  DOM.configModal.classList.add("active");
}

function closeConfigModal() {
  if (!DOM.configModal) return;
  DOM.configModal.classList.remove("active");
}

function initModalListeners() {
  if (DOM.modalCloseBtn) {
    DOM.modalCloseBtn.addEventListener("click", closeConfigModal);
  }
  if (DOM.configModal) {
    DOM.configModal.addEventListener("click", (e) => {
      if (e.target === DOM.configModal) closeConfigModal();
    });
  }

  if (DOM.btnSaveApiUrl) {
    DOM.btnSaveApiUrl.addEventListener("click", () => {
      const newUrl = DOM.apiUrlInput.value.trim().replace(/\/+$/, "");
      if (newUrl) {
        localStorage.setItem(STORAGE_KEY_API_URL, newUrl);
        showToast("API Endpoint Updated", `Now connecting to: ${newUrl}`, "success");
        checkApiHealth(newUrl);
        closeConfigModal();
      }
    });
  }

  if (DOM.btnPingHealth) {
    DOM.btnPingHealth.addEventListener("click", async () => {
      const urlToTest = DOM.apiUrlInput.value.trim().replace(/\/+$/, "");
      DOM.btnPingHealth.disabled = true;
      DOM.btnPingHealth.textContent = "Testing...";

      const result = await checkApiHealth(urlToTest);
      DOM.btnPingHealth.disabled = false;
      DOM.btnPingHealth.textContent = "Test /Health Endpoint";

      DOM.healthTestResult.style.display = "block";
      if (result.ok) {
        DOM.healthTestResult.innerHTML = `
          <div style="color: var(--accent-emerald); font-weight: 600; margin-bottom: 4px;">✓ Health check passed (${result.latency}ms)</div>
          <pre style="font-size: 0.72rem; color: #94a3b8; background: #060911; padding: 6px; border-radius: 4px;">${JSON.stringify(result.data, null, 2)}</pre>
        `;
      } else {
        DOM.healthTestResult.innerHTML = `
          <div style="color: var(--accent-rose); font-weight: 600; margin-bottom: 4px;">✗ Unable to reach FastAPI backend</div>
          <p style="font-size: 0.75rem; color: #94a3b8;">Make sure your FastAPI server is running with CORS enabled. Free-tier Render instances may take ~30-50s to wake up from cold sleep.</p>
        `;
      }
    });
  }
}

// ==============================================================================
// 5. PRESETS & SAMPLE DATA ARCHETYPES
// ==============================================================================
const PRESETS = {
  "high-value": {
    Age: 48,
    Income: 92000,
    Education_Encoded: 2, // Postgraduate / PhD
    Living_With_Encoded: 1, // Partner
    children: 0,
    family_size: 2,
    Recency: 14,
    Customer_Tenure_Days: 520,
    NumWebVisitsMonth: 2,
    total_spend: 1480, // Original raw amount
    total_purchase: 24,
    total_campaigns: 3
  },
  "browsing": {
    Age: 27,
    Income: 26500,
    Education_Encoded: 0, // Basic / Undergraduate
    Living_With_Encoded: 0, // Alone
    children: 1,
    family_size: 2,
    Recency: 68,
    Customer_Tenure_Days: 110,
    NumWebVisitsMonth: 9,
    total_spend: 85, // Original raw amount
    total_purchase: 3,
    total_campaigns: 0
  },
  "balanced": {
    Age: 41,
    Income: 58000,
    Education_Encoded: 1, // Graduate / Master
    Living_With_Encoded: 1, // Partner
    children: 2,
    family_size: 4,
    Recency: 32,
    Customer_Tenure_Days: 390,
    NumWebVisitsMonth: 4,
    total_spend: 620, // Original raw amount
    total_purchase: 15,
    total_campaigns: 1
  }
};

function loadPresetData(presetKey) {
  const data = PRESETS[presetKey];
  if (!data) return;

  DOM.fieldAge.value = data.Age;
  DOM.fieldIncome.value = data.Income;
  DOM.fieldEducation.value = data.Education_Encoded;
  DOM.fieldLivingWith.value = data.Living_With_Encoded;
  DOM.fieldChildren.value = data.children;
  DOM.fieldFamilySize.value = data.family_size;
  DOM.fieldRecency.value = data.Recency;
  DOM.fieldTenure.value = data.Customer_Tenure_Days;
  DOM.fieldWebVisits.value = data.NumWebVisitsMonth;
  DOM.fieldSpend.value = data.total_spend;
  DOM.fieldPurchases.value = data.total_purchase;
  DOM.fieldCampaigns.value = data.total_campaigns;

  // Clear any existing input errors
  clearInputErrors();

  // Update preset active styles
  [DOM.presetHighValue, DOM.presetBrowsing, DOM.presetBalanced].forEach(btn => {
    if (btn) btn.classList.remove("active");
  });

  if (presetKey === "high-value" && DOM.presetHighValue) DOM.presetHighValue.classList.add("active");
  if (presetKey === "browsing" && DOM.presetBrowsing) DOM.presetBrowsing.classList.add("active");
  if (presetKey === "balanced" && DOM.presetBalanced) DOM.presetBalanced.classList.add("active");
}

function initPresets() {
  if (DOM.presetHighValue) {
    DOM.presetHighValue.addEventListener("click", () => loadPresetData("high-value"));
  }
  if (DOM.presetBrowsing) {
    DOM.presetBrowsing.addEventListener("click", () => loadPresetData("browsing"));
  }
  if (DOM.presetBalanced) {
    DOM.presetBalanced.addEventListener("click", () => loadPresetData("balanced"));
  }
}

// ==============================================================================
// 6. FORM VALIDATION & PAYLOAD ASSEMBLY
// ==============================================================================
function clearInputErrors() {
  const inputs = DOM.form.querySelectorAll(".form-input, .form-select");
  inputs.forEach(input => input.classList.remove("input-error"));

  const errorMsgs = DOM.form.querySelectorAll(".field-error-msg");
  errorMsgs.forEach(msg => msg.classList.remove("visible"));
}

function showFieldError(inputElement, message) {
  if (!inputElement) return;
  inputElement.classList.add("input-error");
  
  const parent = inputElement.closest(".input-field-wrapper");
  if (parent) {
    const errorSpan = parent.querySelector(".field-error-msg");
    if (errorSpan) {
      errorSpan.textContent = message;
      errorSpan.classList.add("visible");
    }
  }
}

function validateAndExtractFormData() {
  clearInputErrors();
  let hasError = false;

  // 1. Age: float, ge=18, le=100
  const ageVal = parseFloat(DOM.fieldAge.value);
  if (isNaN(ageVal) || ageVal < 18 || ageVal > 100) {
    showFieldError(DOM.fieldAge, "Age must be between 18 and 100 years.");
    hasError = true;
  }

  // 2. Income: float, ge=0
  const incomeVal = parseFloat(DOM.fieldIncome.value);
  if (isNaN(incomeVal) || incomeVal < 0) {
    showFieldError(DOM.fieldIncome, "Income must be greater than or equal to $0.");
    hasError = true;
  }

  // 3. Education_Encoded: float, ge=0
  const educationVal = parseFloat(DOM.fieldEducation.value);
  if (isNaN(educationVal) || educationVal < 0) {
    showFieldError(DOM.fieldEducation, "Please select an education level.");
    hasError = true;
  }

  // 4. Living_With_Encoded: float, ge=0, le=1 (0=Alone, 1=Partner)
  const livingVal = parseFloat(DOM.fieldLivingWith.value);
  if (isNaN(livingVal) || (livingVal !== 0 && livingVal !== 1)) {
    showFieldError(DOM.fieldLivingWith, "Please select a living arrangement.");
    hasError = true;
  }

  // 5. children: float, ge=0
  const childrenVal = parseFloat(DOM.fieldChildren.value);
  if (isNaN(childrenVal) || childrenVal < 0) {
    showFieldError(DOM.fieldChildren, "Children count must be 0 or greater.");
    hasError = true;
  }

  // 6. family_size: float, ge=1
  const familySizeVal = parseFloat(DOM.fieldFamilySize.value);
  if (isNaN(familySizeVal) || familySizeVal < 1) {
    showFieldError(DOM.fieldFamilySize, "Family size must be at least 1.");
    hasError = true;
  }

  // 7. Recency: float, ge=0
  const recencyVal = parseFloat(DOM.fieldRecency.value);
  if (isNaN(recencyVal) || recencyVal < 0) {
    showFieldError(DOM.fieldRecency, "Recency must be 0 or more days.");
    hasError = true;
  }

  // 8. Customer_Tenure_Days: float, ge=0
  const tenureVal = parseFloat(DOM.fieldTenure.value);
  if (isNaN(tenureVal) || tenureVal < 0) {
    showFieldError(DOM.fieldTenure, "Tenure must be 0 or more days.");
    hasError = true;
  }

  // 9. NumWebVisitsMonth: float, ge=0
  const webVisitsVal = parseFloat(DOM.fieldWebVisits.value);
  if (isNaN(webVisitsVal) || webVisitsVal < 0) {
    showFieldError(DOM.fieldWebVisits, "Web visits must be 0 or greater.");
    hasError = true;
  }

  // 10. total_spend: float, ge=0 (CRITICAL: ORIGINAL value, do NOT apply log1p)
  const spendVal = parseFloat(DOM.fieldSpend.value);
  if (isNaN(spendVal) || spendVal < 0) {
    showFieldError(DOM.fieldSpend, "Total spend must be $0 or greater.");
    hasError = true;
  }

  // 11. total_purchase: float, ge=0
  const purchasesVal = parseFloat(DOM.fieldPurchases.value);
  if (isNaN(purchasesVal) || purchasesVal < 0) {
    showFieldError(DOM.fieldPurchases, "Total purchases must be 0 or greater.");
    hasError = true;
  }

  // 12. total_campaigns: float, ge=0
  const campaignsVal = parseFloat(DOM.fieldCampaigns.value);
  if (isNaN(campaignsVal) || campaignsVal < 0) {
    showFieldError(DOM.fieldCampaigns, "Campaigns accepted must be 0 or greater.");
    hasError = true;
  }

  if (hasError) {
    showToast("Validation Error", "Please check the highlighted form fields.", "error");
    return null;
  }

  // Exact payload matching FastAPI ModelData Pydantic Schema
  return {
    Age: ageVal,
    Income: incomeVal,
    Recency: recencyVal,
    Customer_Tenure_Days: tenureVal,
    total_spend: spendVal,
    total_purchase: purchasesVal,
    total_campaigns: campaignsVal,
    children: childrenVal,
    family_size: familySizeVal,
    Education_Encoded: educationVal,
    Living_With_Encoded: livingVal,
    NumWebVisitsMonth: webVisitsVal
  };
}

// ==============================================================================
// 7. PREDICTION REQUEST & RESPONSE PROCESSING
// ==============================================================================
async function handlePredictSubmit(event) {
  if (event) event.preventDefault();
  if (appState.isPredicting) return;

  const payload = validateAndExtractFormData();
  if (!payload) return;

  const apiUrl = getActiveApiUrl();
  const endpointUrl = `${apiUrl}/Prediction`;

  // Update UI to loading state
  setPredictingState(true);

  console.log(`[CustomerIQ] Dispatching prediction request to: ${endpointUrl}`);
  console.log("[CustomerIQ] Payload (ModelData):", payload);

  const startTime = performance.now();
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 12000); // 12s timeout for ML inference

  try {
    const response = await fetch(endpointUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json"
      },
      body: JSON.stringify(payload),
      signal: controller.signal
    });

    clearTimeout(timeoutId);
    const latencyMs = Math.round(performance.now() - startTime);

    if (!response.ok) {
      let errorDetail = `HTTP ${response.status} ${response.statusText}`;
      try {
        const errorJson = await response.json();
        if (errorJson.detail) {
          if (Array.isArray(errorJson.detail)) {
            // Pydantic 422 validation detail array
            errorDetail = errorJson.detail.map(d => `${d.loc ? d.loc.join('.') : 'field'}: ${d.msg}`).join(', ');
          } else {
            errorDetail = errorJson.detail;
          }
        }
      } catch (e) {
        // Raw text or non-json error
      }
      throw new Error(errorDetail);
    }

    const predictionData = await response.json();
    console.log("[CustomerIQ] Received Prediction Response:", predictionData);

    // Render result card with actual FastAPI response
    renderPredictionResult(predictionData, payload, latencyMs);
    showToast("Segment Identified", `Classified as ${predictionData.segment}`, "success");
    
    // Smooth scroll to result on mobile/tablets
    if (window.innerWidth < 1024 && DOM.resultCard) {
      DOM.resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
    }

  } catch (error) {
    clearTimeout(timeoutId);
    console.error("[CustomerIQ] Prediction error:", error);

    let friendlyMessage = error.message;
    if (error.name === "AbortError") {
      friendlyMessage = "Request timed out. If your Render backend is asleep, please allow 30-50s for cold boot and retry.";
    } else if (error.message.includes("Failed to fetch") || error.message.includes("NetworkError")) {
      friendlyMessage = `Cannot connect to FastAPI at ${apiUrl}. Please verify the Render URL or check CORS configuration in main.py.`;
    }

    showToast("Prediction Failed", friendlyMessage, "error");
    
    // Fallback: If in local preview mode and user wants to preview UI behavior when offline
    showOfflineGuidance(friendlyMessage, apiUrl);

  } finally {
    setPredictingState(false);
  }
}

function setPredictingState(isLoading) {
  appState.isPredicting = isLoading;
  if (!DOM.btnPredict) return;

  DOM.btnPredict.disabled = isLoading;
  if (isLoading) {
    DOM.btnPredict.classList.add("loading");
    DOM.btnPredict.querySelector(".btn-text").textContent = "Analyzing Segments...";
  } else {
    DOM.btnPredict.classList.remove("loading");
    DOM.btnPredict.querySelector(".btn-text").textContent = "Predict Customer Segment";
  }
}

// ==============================================================================
// 8. RESULT RENDERING & SEGMENT INTERPRETATION
// ==============================================================================
function renderPredictionResult(data, inputPayload, latencyMs) {
  const cluster = data.cluster;
  const segmentName = data.segment || `Cluster ${cluster}`;
  const message = data.message || "Customer belongs to a discovered behavioral segment.";

  // Hide placeholder, reveal result card
  if (DOM.placeholderCard) DOM.placeholderCard.style.display = "none";
  if (DOM.resultCard) {
    DOM.resultCard.classList.remove("cluster-0-theme", "cluster-1-theme");
    // Apply cluster specific styling based on response
    if (cluster === 1) {
      DOM.resultCard.classList.add("cluster-1-theme");
    } else {
      DOM.resultCard.classList.add("cluster-0-theme");
    }
    DOM.resultCard.classList.add("active");
  }

  // Update Cluster Badge
  if (DOM.clusterBadge) {
    DOM.clusterBadge.innerHTML = `
      <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:currentColor;"></span>
      Cluster ${cluster}
    `;
  }

  // Update Segment Name
  if (DOM.segmentHeading) {
    DOM.segmentHeading.textContent = segmentName;
  }

  // Update Insight Text
  if (DOM.insightText) {
    DOM.insightText.textContent = message;
  }

  // Generate actionable recommendations based on cluster
  if (DOM.recommendationBullets) {
    if (cluster === 1) {
      DOM.recommendationBullets.innerHTML = `
        <li><strong>VIP Loyalty & Concierge:</strong> Enroll in tier-1 rewards program with bespoke benefits.</li>
        <li><strong>Premium Cross-Selling:</strong> Introduce high-margin product bundles and early product access.</li>
        <li><strong>Retention Defense:</strong> Assign dedicated account manager and priority customer support.</li>
        <li><strong>Exclusive Invitations:</strong> Invite to private VIP webinars and product preview salons.</li>
      `;
    } else {
      DOM.recommendationBullets.innerHTML = `
        <li><strong>Conversion Incentives:</strong> Deploy targeted first-time or limited-time discount vouchers.</li>
        <li><strong>Browse-to-Buy Triggers:</strong> Trigger automated exit-intent and abandoned browse email reminders.</li>
        <li><strong>Affordable Starter Bundles:</strong> Showcase value-packed product packages to lower purchase barrier.</li>
        <li><strong>Personalized Recommendations:</strong> Recommend top-selling products aligned with high web visit history.</li>
      `;
    }
  }

  // Update Snapshot Metrics
  if (DOM.snapshotSpend) DOM.snapshotSpend.textContent = `$${inputPayload.total_spend.toLocaleString()}`;
  if (DOM.snapshotIncome) DOM.snapshotIncome.textContent = `$${inputPayload.Income.toLocaleString()}`;
  if (DOM.snapshotPurchases) DOM.snapshotPurchases.textContent = `${inputPayload.total_purchase} orders`;

  // Update Timestamp
  if (DOM.resultTimestamp) {
    const now = new Date();
    DOM.resultTimestamp.textContent = `Inferred in ${latencyMs}ms at ${now.toLocaleTimeString()}`;
  }

  // Setup Raw Payload Inspector
  if (DOM.rawPayloadView) {
    DOM.rawPayloadView.textContent = JSON.stringify({
      request: {
        endpoint: "/Prediction",
        data: inputPayload
      },
      response: data,
      latency: `${latencyMs}ms`
    }, null, 2);
  }
}

function showOfflineGuidance(errorMessage, apiUrl) {
  // Give helpful diagnostic toast
  showToast("API Unreachable", "Click 'API Status' in top bar to configure your live Render FastAPI URL.", "info");
}

// ==============================================================================
// 9. FORM RESET & EVENT BINDINGS
// ==============================================================================
function initFormEventListeners() {
  if (DOM.form) {
    DOM.form.addEventListener("submit", handlePredictSubmit);
  }

  if (DOM.btnReset) {
    DOM.btnReset.addEventListener("click", () => {
      DOM.form.reset();
      clearInputErrors();
      
      // Reset result card back to placeholder
      if (DOM.resultCard) DOM.resultCard.classList.remove("active");
      if (DOM.placeholderCard) DOM.placeholderCard.style.display = "flex";

      // Reset presets styling
      [DOM.presetHighValue, DOM.presetBrowsing, DOM.presetBalanced].forEach(btn => {
        if (btn) btn.classList.remove("active");
      });

      showToast("Form Reset", "All input fields cleared.", "info");
    });
  }

  // Toggle raw payload inspector
  if (DOM.rawPayloadBtn && DOM.rawPayloadView) {
    DOM.rawPayloadBtn.addEventListener("click", () => {
      DOM.rawPayloadView.classList.toggle("active");
      const isExpanded = DOM.rawPayloadView.classList.contains("active");
      DOM.rawPayloadBtn.querySelector("span").textContent = isExpanded ? "Hide Raw JSON" : "View Raw JSON Payload";
    });
  }

  // Remove error styling on live input typing
  const formInputs = DOM.form.querySelectorAll(".form-input, .form-select");
  formInputs.forEach(input => {
    input.addEventListener("input", () => {
      input.classList.remove("input-error");
      const parent = input.closest(".input-field-wrapper");
      if (parent) {
        const errorSpan = parent.querySelector(".field-error-msg");
        if (errorSpan) errorSpan.classList.remove("visible");
      }
    });
  });
}

// ==============================================================================
// 10. TOAST NOTIFICATION UTILITY
// ==============================================================================
function showToast(title, message, type = "info") {
  if (!DOM.toastContainer) return;

  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;

  const iconSvg = type === "success" 
    ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M20 6L9 17l-5-5"/></svg>`
    : type === "error"
    ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`
    : `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>`;

  toast.innerHTML = `
    <div class="toast-icon">${iconSvg}</div>
    <div class="toast-content">
      <h5>${title}</h5>
      <p>${message}</p>
    </div>
  `;

  DOM.toastContainer.appendChild(toast);

  // Auto dismiss after 4 seconds
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(40px)";
    setTimeout(() => toast.remove(), 300);
  }, 4200);
}

// ==============================================================================
// 11. HERO VISUAL CLUSTER SCATTER GENERATOR
// ==============================================================================
function renderHeroClusterNodes() {
  const svg = document.getElementById("hero-cluster-svg");
  if (!svg) return;

  // Generate synthetic PCA cluster points for visual illustration
  const cluster0Points = [
    { x: 70, y: 190 }, { x: 95, y: 160 }, { x: 120, y: 210 },
    { x: 80, y: 240 }, { x: 130, y: 175 }, { x: 105, y: 225 },
    { x: 60, y: 170 }, { x: 145, y: 195 }, { x: 110, y: 150 },
    { x: 160, y: 220 }, { x: 90, y: 260 }, { x: 135, y: 250 }
  ];

  const cluster1Points = [
    { x: 280, y: 90 }, { x: 310, y: 120 }, { x: 260, y: 70 },
    { x: 340, y: 95 }, { x: 295, y: 140 }, { x: 325, y: 60 },
    { x: 370, y: 110 }, { x: 250, y: 115 }, { x: 355, y: 145 },
    { x: 300, y: 65 }, { x: 275, y: 160 }, { x: 330, y: 165 }
  ];

  // Draw cluster centroids
  const c0Centroid = `<circle cx="105" cy="205" r="14" fill="rgba(245, 158, 11, 0.2)" stroke="#f59e0b" stroke-width="2" stroke-dasharray="3,3" />
                     <circle cx="105" cy="205" r="5" fill="#f59e0b" />`;
  const c1Centroid = `<circle cx="310" cy="105" r="14" fill="rgba(16, 185, 129, 0.2)" stroke="#10b981" stroke-width="2" stroke-dasharray="3,3" />
                     <circle cx="310" cy="105" r="5" fill="#10b981" />`;

  let nodesHtml = `
    <!-- Grid lines -->
    <line x1="40" y1="280" x2="380" y2="280" stroke="rgba(255,255,255,0.1)" stroke-width="1" />
    <line x1="40" y1="40" x2="40" y2="280" stroke="rgba(255,255,255,0.1)" stroke-width="1" />
    <text x="210" y="296" fill="#64748b" font-size="10" text-anchor="middle">PCA Component 1 (Spending & Volume)</text>
    <text x="18" y="160" fill="#64748b" font-size="10" transform="rotate(-90 18 160)" text-anchor="middle">PCA Component 2 (Tenure & Visits)</text>
    ${c0Centroid}
    ${c1Centroid}
  `;

  // Draw nodes
  cluster0Points.forEach((pt, i) => {
    nodesHtml += `<circle class="sim-node" cx="${pt.x}" cy="${pt.y}" r="5" fill="#fbbf24" opacity="0.85">
      <title>Cluster 0 Point #${i + 1} (Browsing Customer)</title>
    </circle>`;
  });

  cluster1Points.forEach((pt, i) => {
    nodesHtml += `<circle class="sim-node" cx="${pt.x}" cy="${pt.y}" r="5" fill="#34d399" opacity="0.85">
      <title>Cluster 1 Point #${i + 1} (High-Value Customer)</title>
    </circle>`;
  });

  svg.innerHTML = nodesHtml;
}

// ==============================================================================
// 12. MOBILE NAVIGATION DRAWER
// ==============================================================================
function initMobileMenu() {
  if (!DOM.mobileMenuBtn || !DOM.mobileNavDrawer) return;

  DOM.mobileMenuBtn.addEventListener("click", () => {
    const isVisible = DOM.mobileNavDrawer.style.display === "flex";
    DOM.mobileNavDrawer.style.display = isVisible ? "none" : "flex";
  });

  const mobileLinks = DOM.mobileNavDrawer.querySelectorAll("a");
  mobileLinks.forEach(link => {
    link.addEventListener("click", () => {
      DOM.mobileNavDrawer.style.display = "none";
    });
  });
}
