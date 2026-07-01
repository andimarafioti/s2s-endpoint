# A ticket + poll queue for session admission when computes are full

When every compute slot is busy, `POST /session` already waits: `acquire()` blocks
on a condition variable up to `allocate_timeout_s` (default 900s), and a `release()`
elsewhere fires `notify_all()`, waking every waiter to race for the freed slot
(`endpoint_pool_router.py`). This works but is invisible and fragile — the caller
holds a single HTTP request open for minutes with no feedback, and that request
dies to browser/proxy idle timeouts long before the 900s budget, silently dropping
the waiter. We keep the internal capacity-wait but replace the *client-facing*
contract with an explicit queue: `POST /session` returns immediately with either a
grant (a slot was free) or a **ticket** (`{queue_id, position}`); the client then
polls `GET /queue/{id}` every ~2s until the ticket is claimed and the poll returns
the `connect_url`. A ticket is dropped if it is not polled within a short TTL (~8s),
which is how an abandoned waiter is detected. The queue is FIFO and tier-blind, and
is bounded (`QUEUE_MAX_DEPTH`, default 100) — a would-be waiter past the cap is told
the queue is at capacity rather than admitted to an unbounded line.

Admission is **pull-driven**: the LB keeps an ordered list of waiting ticket ids,
and only the head ticket may claim a free slot on its next poll. `release()` is
unchanged. Budget is *not* reserved when a ticket is minted; the Space reserves (and
re-checks) the daily limit only at the claim poll — so waiting in line never costs
usage time.

## Considered options

- **Push (reserve-on-release).** `release()` pops the head ticket and reserves the
  freed slot for it immediately; the poll just reads state. Rejected: it reserves a
  slot for a client that may already be gone, tying up a freed compute for the full
  `pending_timeout_s` before the reaper reclaims it. Pull never reserves for a client
  that isn't actively polling, so an abandoned ticket can't hold a slot — the exact
  robustness property we're after. Push's only win is sub-second handoff, which a
  voice demo doesn't need (≤ one poll interval is fine).
- **Keep the blocking `POST /session`, add a position side-channel.** Rejected:
  leaves the fragile minutes-long request in place — the thing we set out to remove.
- **Stream position over SSE/WebSocket instead of polling.** Snappier, but
  reintroduces a long-lived connection through the Space's `/api/session` proxy (the
  failure mode we're eliminating) for a demo that doesn't need live-updating ETAs.
- **Report an estimated wait (ETA), not just position.** Rejected: session length is
  tier-capped and wildly variable, and a cold pool can add minutes of wake latency no
  average predicts. A countdown that stalls or jumps backward reads as broken; a bare
  position only ever decreases.
- **Tier-priority queue (PRO/Team skip the line).** Rejected: PRO/Team already get
  unbounded session *duration*; line-skipping on top can starve free users behind an
  endless stream of PROs. The ordered list leaves this as a cheap later add if wanted.
- **Reserve budget at enqueue.** Rejected: a multi-minute wait would burn an anon
  user's entire 5-minute allowance before they say a word. Reserve at claim instead.

## Consequences

- New client contract, spanning two repos. The LB gains `GET /queue/{id}` (and a
  DELETE/beacon to leave early) alongside the reshaped `POST /session`; the Space
  proxies both (`/api/session` to enqueue or grant, `/api/queue/{id}` to poll) and
  moves `limiter.begin` from the session POST to the claim poll, re-checking
  `remaining` there. Changing the protocol later means coordinating both repos.
- Admission latency is bounded by one poll interval (~2s) after a slot frees, and a
  vanished head-of-line ticket wastes a freed slot only until its TTL lapses (~8s),
  after which the next ticket becomes head. Both are self-healing, no operator action.
- Abandonment has two regimes: **while waiting**, detected by missed polls against the
  ticket TTL (new queue reaper); **after claim but before dialing**, already handled by
  the existing pending-session reaper (`pending_timeout_s`). Only the first is new.
- Queued time is free — only connected conversation time counts against the daily
  limit, which already falls out of counting `connected_at`→disconnect and reserving
  at claim.
- `QUEUE_MAX_DEPTH` (default 100) is a ceiling, not a target, and an env knob rather
  than a code change. Two things bound it: poll load grows as `depth / poll_interval`
  req/s (doubled across the LB and the Space that proxies it), and position only helps
  while the wait is bearable (`depth × avg_session ÷ live_slots`). 100 is headroom that
  keeps the at-capacity modal rare; past ~200, raise `QUEUE_POLL_INTERVAL_S` and make
  the per-poll position lookup O(1) (it is O(depth) today).
