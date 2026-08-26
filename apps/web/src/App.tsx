import { FormEvent, useEffect, useMemo, useState } from "react";
import type { AskResponse } from "./types";

const apiBase = import.meta.env.VITE_API_URL ?? "";

type Turn = {
  query: string;
  response?: AskResponse;
  error?: string;
};

export default function App() {
  const [query, setQuery] = useState("Who is Dongrang?");
  const [configId, setConfigId] = useState("hybrid_rerank_refine");
  const [configs, setConfigs] = useState<string[]>([]);
  const [research, setResearch] = useState(false);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetch(`${apiBase}/v1/configs`)
      .then((r) => r.json())
      .then((data) => setConfigs(data.configs ?? []))
      .catch(() => setConfigs(["hybrid_rerank_refine", "hybrid", "bm25_only"]));
  }, []);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    const text = query.trim();
    if (!text) return;
    setBusy(true);
    setTurns((prev) => [...prev, { query: text }]);
    try {
      const resp = await fetch(`${apiBase}/v1/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: text,
          config_id: configId,
          debug: research,
        }),
      });
      if (!resp.ok) {
        throw new Error(`Ask failed (${resp.status})`);
      }
      const data = (await resp.json()) as AskResponse;
      setTurns((prev) => {
        const copy = [...prev];
        copy[copy.length - 1] = { query: text, response: data };
        return copy;
      });
    } catch (err) {
      setTurns((prev) => {
        const copy = [...prev];
        copy[copy.length - 1] = {
          query: text,
          error: err instanceof Error ? err.message : "Request failed",
        };
        return copy;
      });
    } finally {
      setBusy(false);
    }
  }

  const options = useMemo(
    () => (configs.length ? configs : [configId]),
    [configs, configId],
  );

  return (
    <div className="app">
      <header>
        <h1>Limbus Librarian</h1>
        <p className="disclaimer">
          Independent fan-made lore assistant. Not affiliated with Project Moon.
        </p>
      </header>
      <div className="toolbar">
        <label>
          Retrieval config{" "}
          <select value={configId} onChange={(e) => setConfigId(e.target.value)}>
            {options.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>
        </label>
        <label>
          <input
            type="checkbox"
            checked={research}
            onChange={(e) => setResearch(e.target.checked)}
          />{" "}
          Research mode
        </label>
      </div>
      <form onSubmit={onSubmit}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask about Limbus Company lore"
        />
        <button type="submit" disabled={busy}>
          {busy ? "Searching…" : "Ask"}
        </button>
      </form>
      <div className="chat">
        {turns.map((turn, i) => (
          <article className="bubble" key={`${turn.query}-${i}`}>
            <div className="bubble user">{turn.query}</div>
            {turn.error && <p className="error">{turn.error}</p>}
            {turn.response && (
              <>
                <p>{turn.response.answer}</p>
                <div className="sources">
                  {turn.response.citations.map((cite) => (
                    <div className="source" key={cite.chunk_id}>
                      <strong>{cite.title}</strong> — {cite.section_path}
                      <div>
                        <a href={cite.url} target="_blank" rel="noreferrer">
                          Open source
                        </a>
                      </div>
                      <p>{cite.snippet}</p>
                    </div>
                  ))}
                </div>
                {research && turn.response.trace && (
                  <div className="debug">
                    <p>
                      Combined search used keyword matching and semantic similarity
                      {turn.response.trace.hops ? `, then refined once` : ""}.
                    </p>
                    <p>Config: {turn.response.trace.config_id}</p>
                    <p>Hops: {turn.response.trace.hops}</p>
                    {turn.response.trace.hits.map((hit) => (
                      <div key={hit.chunk_id}>
                        #{hit.rank} {hit.title} ({hit.retriever_name}) score{" "}
                        {hit.score.toFixed(4)}
                        {hit.score_components?.rerank != null
                          ? ` rerank ${hit.score_components.rerank.toFixed(4)}`
                          : ""}
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
          </article>
        ))}
      </div>
      <p className="notice">
        Wiki writing is licensed CC BY-SA 4.0 by the Limbus Company Wiki (wiki.gg).
        Fixture answers in development may use sample documents rather than a live
        crawl.
      </p>
    </div>
  );
}
