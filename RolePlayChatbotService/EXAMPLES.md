# A/B comparison — RAG+GCA off vs on

Same model (CHAI Guanaco), same persona, same prompt. The only difference between
the two columns is whether persona-RAG (UtteranceIndex) and GCA scene retrieval
are enabled.

## mr_darcy

**Prompt:** _Mr. Darcy, your manners at the ball were terribly cold._

### Without RAG/GCA — only `profile_markdown` injected

> My manners, madam, are my own affair. — Though I grant the observation suggests a certain interest in my character.

_(scene_score=None, chunks=0)_

### With RAG + GCA — utterance retrieval + scene match + S·A·T format

> I am not at all apt to be gay, madam. — I am rarely at ease in large assemblies. (His jaw tightens slightly) — Is that a charge you bring against me with satisfaction?

_(scene_score=0.547, chunks=4)_

---

## atticus_finch

**Prompt:** _Atticus, I'm afraid I might lose my temper in court tomorrow._

### Without RAG/GCA — only `profile_markdown` injected

> (leans forward, resting elbows on knees. His eyes are gentle but direct.) You're not alone in that fear, counsel. Every courtroom makes a man question his own soul.
> 
> (Sits back slowly, hands steepled in contemplation. A slight crease appears between his brows as he studies your face.

_(scene_score=None, chunks=0)_

### With RAG + GCA — utterance retrieval + scene match + S·A·T format

> I understand that fear. When you feel that heat rising, remember this: the courtroom is no place for anger. It's a place for facts and fairness. Take a breath, count to ten, and let your preparation speak louder than any emotion.
> 
> That's what I do.

_(scene_score=0.542, chunks=4)_

---

## sherlock_holmes

**Prompt:** _Sherlock, why do you bother helping people who hate you?_

### Without RAG/GCA — only `profile_markdown` injected

> Why, my dear fellow, it is precisely because they hate me that I must help them. Their scorn proves they misunderstand the game

_(scene_score=None, chunks=0)_

### With RAG + GCA — utterance retrieval + scene match + S·A·T format

> "Because hatred is merely a misplaced passion, and passions always leave traces, my dear Watson."

_(scene_score=0.504, chunks=4, action='Stirs tea slowly, eyes fixed on the swirling liquid, a faint smile playing on his lips', thought='The question reveals more about Watson than the client')_

---

## scarlett_ohara

**Prompt:** _Scarlett, the war is here. What will you do?_

### Without RAG/GCA — only `profile_markdown` injected

> I'll do what I've always done—survive. Tara will thrive again, and I'll make sure of it. (My jaw sets, eyes blazing with determination)

_(scene_score=None, chunks=0)_

### With RAG + GCA — utterance retrieval + scene match + S·A·T format

> Fiddle-dee-dee, Rhett! The war is just more talk. Those Yankees wouldn't dare come this far South. They'll be whipped before they get near Tara. Besides, Ashley said

_(scene_score=0.620, chunks=4, action='waves hand dismissively while smoothing her green velvet dress', thought='Please, God, don’t let him see how my hands are shaking')_

---

## Note on Speech / Action / Thought output

The CHAI endpoint enforces `max_output_tokens=80` server-side (see the
smoke test in `chai/client.py`'s docstring). With three required labels
(`Action:` / `Speech:` / `Thought:`) the tightest replies fit; richer
turns frequently get truncated, and **`Thought` is the section most often
dropped** because the model emits Action + Speech first and runs out of
budget.

We accept this — we don't truncate-then-rerun, we don't ask CHAI for a
separate Thought call. When `Thought` is present (~50% of turns in our
smoke runs) the UI shows it under the "show inner" toggle; when absent
the bubble simply has Action + Speech. The parser falls through cleanly:
`parse_sat()` returns `thought=None`, the API surfaces that, and the UI
just doesn't render the empty section.

If you want to maximise Thought retention without paying for a second
CHAI round-trip, swap the order in `prompts/gca.py:SAT_INSTRUCTION` to
`Thought → Action → Speech`. That trades "always-truncated Speech tail"
for "always-present Thought" — different aesthetic; we kept the current
order because Speech feels worse to clip mid-sentence than Thought does.
