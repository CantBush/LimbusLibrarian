export type Citation = {
  chunk_id: string;
  doc_id: string;
  title: string;
  url: string;
  section_path: string;
  snippet: string;
};

export type RetrievalHit = {
  chunk_id: string;
  doc_id: string;
  title: string;
  url: string;
  section_path: string;
  score: number;
  rank: number;
  retriever_name: string;
  score_components: Record<string, number>;
  kept?: boolean | null;
  relevant_score?: number | null;
  text: string;
};

export type AskResponse = {
  answer: string;
  citations: Citation[];
  refused: boolean;
  disclaimer: string;
  trace: {
    query: string;
    config_id: string;
    hops: number;
    refined_queries: string[];
    steps: { name: string; detail: Record<string, unknown> }[];
    hits: RetrievalHit[];
    kept_hits: RetrievalHit[];
  } | null;
};
