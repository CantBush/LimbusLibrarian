const $ = (selector) => document.querySelector(selector);
const form = $("#ask-form");
const query = $("#query");
const config = $("#config");
const answerText = $("#answer-text");
const errorBox = $("#error");
let lastCitations = [];

function escapeHtml(value) {
  const node = document.createElement("div");
  node.textContent = value ?? "";
  return node.innerHTML;
}

function toast(message) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.add("show");
  setTimeout(() => element.classList.remove("show"), 1700);
}

function renderCitations(citations) {
  lastCitations = citations;
  $("#citation-chips").innerHTML = citations.map((citation, index) =>
    `<a class="chip" href="${escapeHtml(citation.url)}" target="_blank" rel="noopener"><b>${index + 1}</b>${escapeHtml(citation.title)}</a>`
  ).join("");
}

function renderSources(citations, hits) {
  const byId = new Map(hits.map((hit) => [hit.chunk_id, hit]));
  const items = citations.map((citation) => ({ ...citation, hit: byId.get(citation.chunk_id) }))
    .concat(hits.filter((hit) => !citations.some((citation) => citation.chunk_id === hit.chunk_id))
      .slice(0, Math.max(0, 5 - citations.length)).map((hit) => ({
        chunk_id: hit.chunk_id, title: hit.title, url: hit.url,
        section_path: hit.section_path, snippet: hit.text, hit
      }))).slice(0, 5);
  $("#source-count").textContent = `${items.length} found`;
  $("#sources").classList.toggle("empty", !items.length);
  $("#sources").innerHTML = items.length ? items.map((item, index) => {
    const score = item.hit?.score ?? 0;
    return `<article class="source-card">
      <div class="source-top"><span class="source-number">${index + 1}</span>
      <a class="source-title" href="${escapeHtml(item.url)}" target="_blank" rel="noopener">${escapeHtml(item.title)}</a>
      <span class="source-score">${score.toFixed(2)}</span></div>
      <p class="source-section">${escapeHtml(item.section_path || "Source passage")}</p>
      <p class="source-snippet">${escapeHtml((item.snippet || "").slice(0, 150))}${(item.snippet || "").length > 150 ? "…" : ""}</p>
    </article>`;
  }).join("") : "No sources were returned.";
}

function renderBars(hits) {
  const values = hits.slice(0, 6);
  const max = Math.max(...values.map((hit) => Math.abs(hit.score)), 0.001);
  $("#bars").classList.toggle("empty", !values.length);
  $("#bars").innerHTML = values.length ? values.map((hit) =>
    `<div class="bar-row"><span class="bar-label" title="${escapeHtml(hit.title)}">${escapeHtml(hit.title)}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${Math.max(4, Math.abs(hit.score) / max * 100)}%"></span></span>
      <span class="bar-score">${hit.score.toFixed(2)}</span></div>`
  ).join("") : "Retrieval scores will appear here.";
}

async function ask(question) {
  const send = $(".send");
  send.disabled = true;
  errorBox.textContent = "";
  answerText.textContent = "Searching the archives…";
  $(".user-message p").textContent = question;
  try {
    const response = await fetch("/v1/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: question, config_id: config.value, debug: true })
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "The search could not be completed.");
    answerText.textContent = body.answer;
    const hits = body.trace?.kept_hits?.length ? body.trace.kept_hits : (body.trace?.hits || []);
    renderCitations(body.citations || []);
    renderSources(body.citations || [], hits);
    renderBars(hits);
    $("#history-row").textContent = `${question} · just now`;
    const historyItem = document.createElement("button");
    historyItem.textContent = question;
    historyItem.addEventListener("click", () => { query.value = question; ask(question); });
    $("#history").prepend(historyItem);
  } catch (error) {
    answerText.textContent = "I could not complete that request.";
    errorBox.textContent = error.message;
  } finally {
    send.disabled = false;
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = query.value.trim();
  if (question) ask(question);
});
query.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});
$("#copy-answer").addEventListener("click", async () => {
  await navigator.clipboard.writeText(answerText.textContent);
  toast("Answer copied");
});
$("#new-chat").addEventListener("click", () => {
  query.value = "";
  answerText.textContent = "Ask a question to search the local lore corpus. Answers are grounded in retrieved sources and include citations.";
  renderCitations([]);
  renderSources([], []);
  renderBars([]);
  query.focus();
});
document.querySelectorAll("[data-stub]").forEach((item) => item.addEventListener("click", () => toast("Coming later")));
document.querySelectorAll(".mode-toggle button").forEach((button) => button.addEventListener("click", () => {
  config.value = button.dataset.mode;
  document.querySelectorAll(".mode-toggle button").forEach((item) => item.classList.toggle("active", item === button));
}));
config.addEventListener("change", () => {
  document.querySelectorAll(".mode-toggle button").forEach((item) => item.classList.toggle("active", item.dataset.mode === config.value));
});

fetch("/v1/health").then((response) => response.json()).then((health) => {
  $(".status-dot").classList.add("ready");
  $("#llm-status").textContent = health.llm_configured ? "OpenAI ready" : "Local fallback · add key in .env";
}).catch(() => { $("#llm-status").textContent = "API unavailable"; });
