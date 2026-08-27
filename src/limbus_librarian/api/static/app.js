const $ = (selector) => document.querySelector(selector);
const form = $("#ask-form");
const query = $("#query");
const config = $("#config");
const answerText = $("#answer-text");
const errorBox = $("#error");
let lastCitations = [];
const STORAGE_KEY = "limbus-librarian.sessions.v1";
const ACTIVE_KEY = "limbus-librarian.active-session.v1";
let sessions = loadSessions();
let activeSessionId = localStorage.getItem(ACTIVE_KEY);
let catalogState = { mode: "explore", page: 1, query: "" };
let askStatusTimer;
let viewTransitionToken = 0;

function prefersReducedMotion() {
  return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function loadSessions() {
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    return Array.isArray(stored) ? stored : [];
  } catch {
    return [];
  }
}

function newSession() {
  return { id: crypto.randomUUID?.() || `${Date.now()}-${Math.random()}`, title: "New chat", turns: [], updatedAt: Date.now() };
}

function currentSession() {
  let session = sessions.find((item) => item.id === activeSessionId);
  if (!session) {
    session = newSession();
    sessions.unshift(session);
    activeSessionId = session.id;
  }
  return session;
}

function saveSessions() {
  sessions = sessions.slice(0, 30).map((session) => ({ ...session, turns: session.turns.slice(-50) }));
  localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
  localStorage.setItem(ACTIVE_KEY, activeSessionId);
}

function escapeHtml(value) {
  const node = document.createElement("div");
  node.textContent = value ?? "";
  return node.innerHTML;
}

function escapeAttribute(value) {
  return escapeHtml(value).replaceAll('"', "&quot;");
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

function renderAnswer(answer, citations) {
  answerText.replaceChildren();
  const parts = String(answer ?? "").split(/(\[\d+\])/g);
  parts.forEach((part) => {
    const match = part.match(/^\[(\d+)\]$/);
    if (!match) {
      answerText.append(document.createTextNode(part));
      return;
    }
    const number = Number(match[1]);
    const citation = citations[number - 1];
    if (!citation) {
      answerText.append(document.createTextNode(part));
      return;
    }
    const link = document.createElement("a");
    link.className = "inline-citation";
    link.href = citation.url;
    link.target = "_blank";
    link.rel = "noopener";
    link.setAttribute("aria-label", `Source ${number}: ${citation.title}`);
    link.textContent = `[${number}]`;
    link.addEventListener("click", () => {
      const card = document.getElementById(`source-${number}`);
      if (!card) return;
      document.querySelectorAll(".source-card.highlight").forEach((item) => item.classList.remove("highlight"));
      card.classList.add("highlight");
      card.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "center" });
      setTimeout(() => card.classList.remove("highlight"), 1800);
    });
    const sup = document.createElement("sup");
    sup.append(link);
    answerText.append(sup);
  });
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
    return `<article class="source-card" id="source-${index + 1}">
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
  $("#bars").innerHTML = values.length ? values.map((hit) => {
    const graphScore = hit.score_components?.graph;
    const graphLabel = graphScore == null ? "" : ` · graph ${graphScore.toFixed(2)}`;
    return `<div class="bar-row"><span class="bar-label" title="${escapeHtml(hit.title)}">${escapeHtml(hit.title)}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${Math.max(4, Math.abs(hit.score) / max * 100)}%"></span></span>
      <span class="bar-score">${hit.score.toFixed(2)}${graphLabel}</span></div>`;
  }).join("") : "Retrieval scores will appear here.";
}

function renderSessionList() {
  const history = $("#history");
  history.innerHTML = sessions.length ? sessions.map((session) =>
    `<button data-session-id="${escapeAttribute(session.id)}" class="${session.id === activeSessionId ? "active" : ""}">${escapeHtml(session.title)}</button>`
  ).join("") : '<span class="history-empty">No saved chats yet.</span>';
  history.querySelectorAll("[data-session-id]").forEach((button) => button.addEventListener("click", () => {
    activeSessionId = button.dataset.sessionId;
    saveSessions();
    restoreSession();
  }));
}

function restoreSession() {
  const session = currentSession();
  const turn = session.turns.at(-1);
  $(".user-message p").textContent = turn?.query || "Start a new lore question.";
  if (turn?.answer) {
    renderAnswer(turn.answer, turn.citations || []);
    renderCitations(turn.citations || []);
    renderSources(turn.citations || [], turn.hits || []);
    renderBars(turn.hits || []);
    $("#history-row").textContent = `${session.turns.length} question${session.turns.length === 1 ? "" : "s"} saved locally`;
  } else {
    answerText.textContent = "Ask a question to search the local lore corpus. Answers are grounded in retrieved sources and include citations.";
    renderCitations([]);
    renderSources([], []);
    renderBars([]);
    $("#history-row").textContent = "No questions asked yet in this session.";
  }
  renderSessionList();
}

function setAskLoading() {
  const status = $("#ask-status");
  const states = ["Searching…", "Reading sources…", "Writing…"];
  let stateIndex = 0;
  status.textContent = states[stateIndex];
  status.classList.remove("hidden");
  const surfaces = [$(".answer-card"), $("#sources").closest(".panel"), $("#bars").closest(".panel")];
  surfaces.forEach((surface) => {
    surface.classList.add("is-loading");
    surface.setAttribute("aria-busy", "true");
  });
  answerText.innerHTML = '<span class="skeleton-stack" aria-hidden="true"><span class="skeleton-line"></span><span class="skeleton-line"></span><span class="skeleton-line"></span></span>';
  $("#citation-chips").innerHTML = '<span class="skeleton-chip" aria-hidden="true"></span>';
  $("#source-count").textContent = "Searching";
  $("#sources").classList.remove("empty");
  $("#sources").innerHTML = '<span aria-hidden="true"><span class="skeleton-source"></span><span class="skeleton-source"></span><span class="skeleton-source"></span></span>';
  $("#bars").classList.remove("empty");
  $("#bars").innerHTML = '<span aria-hidden="true"><span class="skeleton-bar"></span><span class="skeleton-bar"></span><span class="skeleton-bar"></span></span>';
  askStatusTimer = setInterval(() => {
    stateIndex = Math.min(stateIndex + 1, states.length - 1);
    status.textContent = states[stateIndex];
    if (stateIndex === states.length - 1) clearInterval(askStatusTimer);
  }, 1100);
}

function stopAskLoading() {
  clearInterval(askStatusTimer);
  askStatusTimer = undefined;
  const status = $("#ask-status");
  status.classList.add("hidden");
  status.textContent = "";
  [$(".answer-card"), $("#sources").closest(".panel"), $("#bars").closest(".panel")].forEach((surface) => {
    surface.classList.remove("is-loading");
    surface.removeAttribute("aria-busy");
  });
}

function fadeInResults() {
  [answerText, $("#citation-chips"), $("#sources"), $("#bars")].forEach((element) => {
    element.classList.remove("results-enter");
    void element.offsetWidth;
    element.classList.add("results-enter");
    setTimeout(() => element.classList.remove("results-enter"), 280);
  });
}

async function ask(question) {
  const send = $(".send");
  const session = currentSession();
  const recentHistory = session.turns.slice(-4).map((turn) => turn.query);
  const documentTypes = [...document.querySelectorAll(".type-filters input:checked")].map((input) => input.value);
  const maxCanto = $("#max-canto").value;
  send.disabled = true;
  errorBox.textContent = "";
  $(".user-message p").textContent = question;
  setAskLoading();
  try {
    const response = await fetch("/v1/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: question,
        config_id: config.value,
        debug: true,
        document_types: documentTypes,
        max_canto: maxCanto ? Number(maxCanto) : null,
        history: recentHistory
      })
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "The search could not be completed.");
    const hits = body.trace?.kept_hits?.length ? body.trace.kept_hits : (body.trace?.hits || []);
    renderAnswer(body.answer, body.citations || []);
    renderCitations(body.citations || []);
    renderSources(body.citations || [], hits);
    renderBars(hits);
    fadeInResults();
    session.turns.push({ query: question, answer: body.answer, citations: body.citations || [], hits, createdAt: Date.now() });
    session.title = session.turns[0].query.slice(0, 65);
    session.updatedAt = Date.now();
    sessions.sort((left, right) => right.updatedAt - left.updatedAt);
    saveSessions();
    renderSessionList();
    $("#history-row").textContent = `${session.turns.length} question${session.turns.length === 1 ? "" : "s"} saved locally`;
  } catch (error) {
    restoreSession();
    errorBox.textContent = error.message;
  } finally {
    stopAskLoading();
    send.disabled = false;
  }
}

const catalogModes = {
  explore: { title: "Explore", description: "Browse every lore page in the local retrieval corpus.", types: "" },
  characters: { title: "Characters", description: "Browse characters and the twelve Sinners.", types: "character,sinner" },
  glossary: { title: "Glossary", description: "Browse world concepts, factions, abnormalities, locations, and events.", types: "world,faction,abnormality,location,event" }
};

async function transitionTo(target, updateContent) {
  const panels = [...document.querySelectorAll(".view-panel")];
  const current = panels.find((panel) => !panel.classList.contains("hidden"));
  const refreshCurrent = current === target && updateContent;
  if (current === target && !refreshCurrent) return;
  const token = ++viewTransitionToken;
  panels.forEach((panel) => panel.classList.remove("view-enter", "view-exit"));
  if (!current || prefersReducedMotion()) {
    if (updateContent) await updateContent();
    panels.forEach((panel) => panel.classList.toggle("hidden", panel !== target));
    return;
  }
  current.classList.add("view-exit");
  await delay(180);
  if (token !== viewTransitionToken) return;
  if (!refreshCurrent) current.classList.add("hidden");
  current.classList.remove("view-exit");
  if (updateContent) await updateContent();
  if (token !== viewTransitionToken) return;
  if (!refreshCurrent) target.classList.remove("hidden");
  target.classList.add("view-enter");
  await delay(240);
  if (token === viewTransitionToken) target.classList.remove("view-enter");
}

function showView(name) {
  const target = name === "about" ? $("#about-view") : name === "chat" ? $("#chat-view") : $("#catalog-view");
  document.querySelectorAll("header [data-view]").forEach((link) => {
    const active = link.dataset.view === name;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  const updateCatalog = catalogModes[name] ? async () => {
    catalogState = { mode: name, page: 1, query: "" };
    $("#catalog-search").value = "";
    await loadCatalog();
  } : undefined;
  return transitionTo(target, updateCatalog);
}

async function loadCatalog() {
  const mode = catalogModes[catalogState.mode];
  $("#catalog-title").textContent = mode.title;
  $("#catalog-description").textContent = mode.description;
  $("#catalog-results").innerHTML = '<div class="catalog-empty">Loading local catalog…</div>';
  const params = new URLSearchParams({ page: String(catalogState.page), per_page: "24" });
  if (mode.types) params.set("type", mode.types);
  if (catalogState.query) params.set("q", catalogState.query);
  try {
    const response = await fetch(`/v1/documents?${params}`);
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "Catalog could not be loaded.");
    renderCatalog(body);
  } catch (error) {
    $("#catalog-results").innerHTML = `<div class="catalog-empty">${escapeHtml(error.message)}</div>`;
    $("#catalog-pagination").replaceChildren();
  }
}

function renderCatalog(body) {
  const results = $("#catalog-results");
  results.innerHTML = body.items.length ? body.items.map((item) =>
    `<article class="catalog-card">
      <span class="catalog-type">${escapeHtml(item.document_type.replaceAll("_", " "))}</span>
      <h2>${escapeHtml(item.title)}</h2>
      <p>${escapeHtml(item.summary)}</p>
      <p class="catalog-cantos">${escapeHtml(item.cantos.join(" · ") || "General lore")}</p>
      <div class="catalog-actions">
        <button class="text-button" data-open-doc="${escapeAttribute(item.doc_id)}">Read excerpt</button>
        <button class="text-button" data-ask-title="${escapeAttribute(item.title)}">Ask about this</button>
      </div>
    </article>`
  ).join("") : '<div class="catalog-empty">No matching pages in the loaded corpus.</div>';
  results.querySelectorAll("[data-open-doc]").forEach((button) => button.addEventListener("click", () => openDocument(button.dataset.openDoc)));
  results.querySelectorAll("[data-ask-title]").forEach((button) => button.addEventListener("click", () => {
    query.value = `Who is ${button.dataset.askTitle}?`;
    showView("chat").then(() => query.focus());
  }));
  const pagination = $("#catalog-pagination");
  pagination.innerHTML = body.pages > 1
    ? `<button data-page="${body.page - 1}" ${body.page <= 1 ? "disabled" : ""}>Previous</button>
       <span>Page ${body.page} of ${body.pages} · ${body.total} pages</span>
       <button data-page="${body.page + 1}" ${body.page >= body.pages ? "disabled" : ""}>Next</button>`
    : `<span>${body.total} page${body.total === 1 ? "" : "s"}</span>`;
  pagination.querySelectorAll("[data-page]:not(:disabled)").forEach((button) => button.addEventListener("click", () => {
    catalogState.page = Number(button.dataset.page);
    loadCatalog();
    window.scrollTo({ top: 0, behavior: prefersReducedMotion() ? "auto" : "smooth" });
  }));
}

async function openDocument(docId) {
  try {
    const response = await fetch(`/v1/documents/${encodeURIComponent(docId)}`);
    const item = await response.json();
    if (!response.ok) throw new Error(item.detail || "Document could not be loaded.");
    $("#document-detail").innerHTML = `
      <span class="catalog-type">${escapeHtml(item.document_type.replaceAll("_", " "))}</span>
      <h1>${escapeHtml(item.title)}</h1>
      <p>${escapeHtml(item.summary)}</p>
      <div class="document-sections">${item.sections.map((section) => `<span>${escapeHtml(section)}</span>`).join("")}</div>
      <section class="related-pages">
        <h2>Related pages</h2>
        <div>${(item.related || []).length ? item.related.map((related) =>
          `<button class="text-button" data-related-doc="${escapeAttribute(related.doc_id)}">${escapeHtml(related.title)}</button>`
        ).join("") : "<span>No graph-linked pages in this corpus.</span>"}</div>
      </section>
      <p><a href="${escapeAttribute(item.url)}" target="_blank" rel="noopener">Open attributed wiki page ↗</a></p>
      <small>${escapeHtml(item.attribution_text)} ${escapeHtml(item.license)}</small>`;
    $("#document-detail").querySelectorAll("[data-related-doc]").forEach((button) =>
      button.addEventListener("click", () => openDocument(button.dataset.relatedDoc))
    );
    transitionTo($("#document-view"));
  } catch (error) {
    toast(error.message);
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = query.value.trim();
  if (!question) return;
  query.value = "";
  query.focus();
  ask(question);
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
  const session = newSession();
  sessions.unshift(session);
  activeSessionId = session.id;
  saveSessions();
  query.value = "";
  restoreSession();
  showView("chat").then(() => query.focus());
});
document.querySelectorAll("header [data-view]").forEach((item) => item.addEventListener("click", () => showView(item.dataset.view)));
$("#document-back").addEventListener("click", () => {
  transitionTo($("#catalog-view"));
});
let catalogSearchTimer;
$("#catalog-search").addEventListener("input", () => {
  clearTimeout(catalogSearchTimer);
  catalogSearchTimer = setTimeout(() => {
    catalogState.query = $("#catalog-search").value.trim();
    catalogState.page = 1;
    loadCatalog();
  }, 250);
});
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

currentSession();
saveSessions();
restoreSession();
