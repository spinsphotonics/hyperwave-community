STATUS: BLOCKED: gated by Cloudflare Turnstile human-check
TICKET: F-wycliffe-sil
ROLE: Fetcher
INPUTS READ: sources.csv rows `wheaton-jimelliot-archives`, `wheaton-elisabethelliot-archives`

# Fetch request — Archives of Wheaton College finding aids for the Elliot papers

**URLs:**
- Jim Elliot Collection: https://archives.wheaton.edu/repositories/4/resources/466
- Elisabeth Elliot Papers: https://archives.wheaton.edu/repositories/4/resources/484

**What was tried:** `curl -sS -L --max-time 60 "<url>"` returned a 1000-byte HTML page titled "Human Check" — a Cloudflare Turnstile interactive challenge — instead of the finding aid content, for both URLs. This matches the finding already recorded in the `wheaton-college` chapter's own coverage note this session ("archives.wheaton.edu were unreachable this session (Cloudflare gates)").

**Request procedure:** requires a human using a real browser (Turnstile cannot be solved by an automated `curl` fetch) to open the finding aid pages directly, or to contact the Archives via their public request form for the box/folder list.

**What this would give:** the roster's special step for `wycliffe-sil` calls for "the archive finding aid for the Elliot papers." These collections likely hold correspondence or notes from Jim Elliot's 1950 SIL/Camp Wycliffe summer that would let a Fetcher confirm the exact year and location beyond what `eef-journals-1948.txt` (which states 1948) and Wikipedia (which states summer 1950, citing *Shadow of the Almighty*) currently give — see `discrepancies.md`.
