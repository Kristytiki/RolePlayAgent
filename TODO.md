# TODO

## Memory persistence (phase 2)

The backend already persists every chat session to `.sessions/session_<id>/`
via Strands' `FileSessionManager`. The remaining gaps are on the UI side
plus a registry rehydrate.

### 1. Survive page reload — same browser, same user

- [ ] On chat start, write `{persona_id, session_id}` to `localStorage` keyed
      by persona_id. On app boot, hydrate the `byPersona` cache from
      `localStorage` and call `GET /chat/sessions/{id}/history` to repopulate
      bubbles from disk.
- [ ] Backend: when `GET /chat/sessions/{id}/history` is hit and the
      session_id is on disk under `.sessions/` but not in the in-memory
      `SessionRegistry`, rehydrate the Strands `Agent` lazily via
      `build_chat_agent(persona, ..., session_id=...)` (the FileSessionManager
      will auto-restore `agent.messages`). Currently `SessionRegistry.get()`
      only checks the in-memory dict, so after a uvicorn restart the
      session_id 404s even though its messages are still on disk.

### 2. Per-user identity across browsers

- [ ] Move from anonymous "Reader" to a real `user_id` (cookie or header).
      `byPersona` becomes `byUser → byPersona → session`. Backend filters
      `.sessions/` by user.

### 3. "Forget this conversation" button

- [ ] In chat header, add a trash-can button next to "show inner". Clicking
      calls `DELETE /chat/sessions/{id}`, removes the persona from the local
      cache, and starts a fresh greeting on next entry.
- [ ] Backend: `DELETE /chat/sessions/{id}` already removes from the
      registry, but **does not delete the on-disk Strands session directory**.
      Add `shutil.rmtree(".sessions/session_<id>")` to the delete handler.

### 4. Roll back N turns / edit-and-resend

- [ ] Per-bubble "↶ rewind from here" button on bot bubbles. Clicking
      truncates `agent.messages` back to the message before the rewound
      one and persists the trimmed state to disk. UI drops bubbles past
      that point. Equivalent to a tactical undo for "I didn't like that
      reply" without nuking the whole session.
- [ ] User-side bubble: "✎ edit" — same truncation but also re-sends the
      edited user message, replacing the original turn with new content.
- [ ] Backend: new endpoint `POST /chat/sessions/{id}/truncate` taking
      `{keep_first: int}` that calls into Strands' SessionManager
      `update_message` / `delete_message` lifecycle hooks (these already
      exist on `FileSessionManager`).

## Animated avatar — render the bot's `Action` line as motion

Inspired by Andrew Ng's DeepLearning.AI short courses on AI avatars
(e.g. "Building Talking Avatars" / Synthesia / D-ID-style talking heads),
we already extract a structured `action` field per turn ("Raises an
eyebrow with a flicker of curiosity"). That string is a free animation
prompt.

- [ ] Wire the parsed `action` into a small avatar component that maps
      action keywords → Lottie / sprite frames (or kicks off a real-time
      talking-head request with the persona portrait + the `speech`
      audio from a TTS pass).
- [ ] Cheaper interim: CSS-driven idle states (eyebrow tilt, lean in,
      eyes narrow) keyed off a tiny enum the parser produces alongside
      the free-form action text.
- [ ] Not a fit for every persona portrait we currently use (some are
      book covers, not faces), so this requires switching the
      Hermione / Heathcliff / Anna stand-ins to actual face crops first.

## Streaming UX

The upstream CHAI endpoint (`POST /endpoints/onsite/chat`) returns the
full `model_output` string in a single response — it does not support SSE
or chunked transfer. So **true streaming is not achievable** at the model
layer; the bot's whole reply is materialised in one CHAI round-trip
(~3-5 s end-to-end).

What we *do* have:

- A typing indicator in the chat composer while waiting on the round-trip
  (3 dots animation, see `App.css :: .bubble.pending .typing`).
- `ChaiModel.stream()` already emits Strands' `messageStart →
  contentBlockDelta → messageStop` event sequence (one big delta), so any
  Strands-Agent-side streaming hook works without further changes.

What's left:

- [ ] Cosmetic typewriter effect on the UI: after the reply lands,
      reveal it character-by-character (~10 ms/char) instead of in one
      jump. Doesn't reduce real latency — purely UX. Implement in
      `App.tsx` by replacing the immediate `setBubbles` with an interval
      that grows the bubble's `message` field.
- [ ] Optional: convert `POST /chat/sessions/{id}/messages` to SSE that
      delivers the same characters slowly server-side. Adds complexity
      (SSE plumbing, EventSource on the client) for the same effect as
      the cosmetic version above. Recommend skipping unless the model
      provider grows real streaming.

## Eval harness

- [ ] Wire `ANTHROPIC_API_KEY` and run the penalty-based judge over the 8
      filtered cases × 4 dimensions, fill in `RESULTS.md` with mean/std.

## Datasets

- [ ] Optional: hand-write `profile_markdown` for the 8 ChatHaruhi-only
      personas (Haruhi, 乔峰, 韦小宝, 王语嫣, 佟湘玉, 胡桃, 钟离, Sheldon)
      to bring the persona library from 9 to 17.
