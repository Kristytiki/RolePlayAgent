// Typed wrappers around the FastAPI backend.

const BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

export type Persona = {
  id: string;
  name: string;
  source: string;
  avatar_emoji: string;
  tagline: string;
  tags: string[];
};

export type CreateSessionResp = {
  session_id: string;
  persona: Persona;
  greeting: string;
};

export type SendMessageResp = {
  reply: string;
  speech: string;
  action: string | null;
  thought: string | null;
  raw: string | null;
  retrieved_scene_score: number | null;
  retrieved_chunk_count: number;
};

export type HistoryTurn = {
  sender: string;
  message: string;
};

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!r.ok) throw new Error(`${r.status}: ${await r.text().catch(() => "")}`);
  return r.json() as Promise<T>;
}

export const api = {
  listPersonas: () => jsonFetch<Persona[]>(`${BASE}/personas`),

  searchPersonas: (query: string, k = 5) =>
    jsonFetch<{ query: string; results: { persona: Persona; score: number }[] }>(
      `${BASE}/personas/search`,
      { method: "POST", body: JSON.stringify({ query, k }) },
    ),

  createSession: (persona_id: string, user_name: string) =>
    jsonFetch<CreateSessionResp>(`${BASE}/chat/sessions`, {
      method: "POST",
      body: JSON.stringify({ persona_id, user_name }),
    }),

  sendMessage: (session_id: string, message: string) =>
    jsonFetch<SendMessageResp>(`${BASE}/chat/sessions/${session_id}/messages`, {
      method: "POST",
      body: JSON.stringify({ message }),
    }),

  getHistory: (session_id: string) =>
    jsonFetch<{ session_id: string; persona_id: string; turns: HistoryTurn[] }>(
      `${BASE}/chat/sessions/${session_id}/history`,
    ),

  deleteSession: (session_id: string) =>
    fetch(`${BASE}/chat/sessions/${session_id}`, { method: "DELETE" }),
};
