import { useEffect, useRef, useState } from "react";
import {
  api,
  type CreateSessionResp,
  type Persona,
  type SendMessageResp,
} from "./api";
import "./App.css";

type Bubble = {
  sender: string;
  message: string;
  action?: string | null;
  thought?: string | null;
  scene_score?: number | null;
  chunks?: number;
};

export default function App() {
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [filtered, setFiltered] = useState<Persona[]>([]);
  const [search, setSearch] = useState("");
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [active, setActive] = useState<CreateSessionResp | null>(null);
  const [bubbles, setBubbles] = useState<Bubble[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [showInner, setShowInner] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.listPersonas().then(setPersonas).catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [bubbles, sending]);

  async function runSearch(q: string) {
    setSearch(q);
    if (!q.trim()) {
      setFiltered([]);
      return;
    }
    setSearching(true);
    try {
      const r = await api.searchPersonas(q, 9);
      setFiltered(r.results.map((x) => x.persona));
    } catch (e) {
      setError(String(e));
    } finally {
      setSearching(false);
    }
  }

  async function startChat(p: Persona) {
    setError(null);
    try {
      const sess = await api.createSession(p.id, "Reader");
      setActive(sess);
      setBubbles([{ sender: p.name, message: sess.greeting }]);
    } catch (e) {
      setError(String(e));
    }
  }

  async function send() {
    if (!active || !input.trim() || sending) return;
    const msg = input.trim();
    setInput("");
    setBubbles((b) => [...b, { sender: "Reader", message: msg }]);
    setSending(true);
    try {
      const r: SendMessageResp = await api.sendMessage(active.session_id, msg);
      setBubbles((b) => [
        ...b,
        {
          sender: active.persona.name,
          message: r.speech || r.reply,
          action: r.action,
          thought: r.thought,
          scene_score: r.retrieved_scene_score,
          chunks: r.retrieved_chunk_count,
        },
      ]);
    } catch (e) {
      setError(String(e));
      setBubbles((b) => [...b, { sender: "system", message: `[error: ${String(e)}]` }]);
    } finally {
      setSending(false);
    }
  }

  function endChat() {
    if (active) api.deleteSession(active.session_id).catch(() => {});
    setActive(null);
    setBubbles([]);
    setInput("");
  }

  if (active) {
    return (
      <div className="page chat-page">
        <header className="chat-header">
          <button className="back" onClick={endChat}>← back</button>
          <div className="chat-title">
            <span className="emoji">{active.persona.avatar_emoji}</span>
            <div>
              <div className="name">{active.persona.name}</div>
              <div className="source">{active.persona.source}</div>
            </div>
          </div>
          <label className="toggle">
            <input
              type="checkbox"
              checked={showInner}
              onChange={(e) => setShowInner(e.target.checked)}
            />
            <span>show inner</span>
          </label>
        </header>

        <div className="bubbles" ref={scrollRef}>
          {bubbles.map((b, i) => (
            <div key={i} className={`bubble ${b.sender === "Reader" ? "user" : "bot"}`}>
              <div className="meta">{b.sender}</div>
              {b.action && showInner && <div className="action">({b.action})</div>}
              <div className="speech">{b.message}</div>
              {b.thought && showInner && <div className="thought">[ {b.thought} ]</div>}
              {b.sender !== "Reader" && b.scene_score != null && showInner && (
                <div className="signal">
                  scene {b.scene_score.toFixed(2)} · chunks {b.chunks ?? 0}
                </div>
              )}
            </div>
          ))}
          {sending && (
            <div className="bubble bot pending">
              <div className="meta">{active.persona.name}</div>
              <div className="speech typing"><span /><span /><span /></div>
            </div>
          )}
        </div>

        <div className="composer">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            placeholder={`Message ${active.persona.name}…  (Enter to send, Shift+Enter for newline)`}
            disabled={sending}
            rows={2}
          />
          <button onClick={send} disabled={sending || !input.trim()} className="send">
            send
          </button>
        </div>

        {error && <div className="banner-error">{error}</div>}
      </div>
    );
  }

  const list = filtered.length > 0 ? filtered : personas;
  return (
    <div className="page picker-page">
      <header className="picker-header">
        <h1>Choose a character</h1>
        <p>Talk to one of {personas.length} literary personas, all sourced from CoSER.</p>
        <div className="search">
          <input
            value={search}
            onChange={(e) => runSearch(e.target.value)}
            placeholder='describe who you want to talk to (e.g. "a brilliant overachiever")'
          />
          {searching && <span className="hint">searching…</span>}
          {filtered.length > 0 && (
            <button className="clear" onClick={() => runSearch("")}>clear</button>
          )}
        </div>
      </header>
      <div className="grid">
        {list.map((p) => (
          <button key={p.id} className="card" onClick={() => startChat(p)}>
            <div className="card-emoji">{p.avatar_emoji}</div>
            <div className="card-name">{p.name}</div>
            <div className="card-tagline">{p.tagline}</div>
            <div className="card-source">{p.source}</div>
            <div className="card-tags">
              {p.tags.slice(0, 4).map((t) => (
                <span key={t} className="tag">{t}</span>
              ))}
            </div>
          </button>
        ))}
      </div>
      {error && <div className="banner-error">{error}</div>}
    </div>
  );
}
