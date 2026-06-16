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

## Eval harness

- [ ] Wire `ANTHROPIC_API_KEY` and run the penalty-based judge over the 8
      filtered cases × 4 dimensions, fill in `RESULTS.md` with mean/std.

## Datasets

- [ ] Optional: hand-write `profile_markdown` for the 8 ChatHaruhi-only
      personas (Haruhi, 乔峰, 韦小宝, 王语嫣, 佟湘玉, 胡桃, 钟离, Sheldon)
      to bring the persona library from 9 to 17.
