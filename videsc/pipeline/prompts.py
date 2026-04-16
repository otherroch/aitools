"""Prompt templates for the Gemma 4 segment-consolidation pipeline.

Adapted from the design document in issue #33.  The pipeline has three stages:

1. **Segment-level** – Analysed per video chunk; produces structured JSON with
   events, objects, actions, scene description and a one-sentence summary.
2. **Window aggregation** – Merges consecutive segment outputs into larger
   windows, deduplicating and preserving chronological order.
3. **Final summary** – Global reduce that produces a coherent high-level
   description with OVERVIEW, TIMELINE, ENTITIES, ACTIONS and THEMES sections.

Each constant holds the *instruction* portion of the prompt.  Callers append
the actual segment / window data before sending to the model.
"""

# ── Stage 1: per-segment prompt ──────────────────────────────────────────
SEGMENT_PROMPT = """\
You are analyzing a {chunk_duration}-second segment of a video.

Your job is to extract factual, observable information only.

RULES:
- Do NOT speculate about intent or emotions unless visually obvious
- Do NOT repeat the same object/event multiple times
- Use short, atomic phrases
- Be consistent in naming (e.g., "man", "woman", "car")

OUTPUT FORMAT (STRICT JSON):
{{
  "timestamp_start": "<HH:MM:SS>",
  "timestamp_end": "<HH:MM:SS>",
  "events": ["..."],
  "objects": ["..."],
  "actions": ["..."],
  "scene": "<short description of environment>",
  "summary": "<1 sentence max>"
}}

EXAMPLE STYLE:
- events: ["man enters room", "sits on chair"]
- objects: ["chair", "table", "laptop"]
- actions: ["walking", "sitting"]

Now analyze the segment.\
"""

# ── Stage 2: window aggregation prompt ───────────────────────────────────
WINDOW_AGGREGATION_PROMPT = """\
You are summarizing a sequence of video segment descriptions.

INPUT:
A list of JSON objects describing consecutive video segments.

TASK:
- Merge duplicate or repeated events
- Preserve chronological order
- Consolidate objects and actions
- Remove noise and trivial details

RULES:
- Do NOT repeat the same event unless new information is added
- Prefer generalization when events repeat (e.g., "person walks around room")
- Keep temporal flow clear

OUTPUT FORMAT (STRICT JSON):
{
  "time_range": "<start–end>",
  "key_events": ["ordered list of important events"],
  "main_objects": ["deduplicated list"],
  "main_actions": ["deduplicated list"],
  "scene_progression": "<how the scene evolves>",
  "summary": "<3–5 sentences>"
}\
"""

# ── Stage 3: final summary prompt ────────────────────────────────────────
FINAL_SUMMARY_PROMPT = """\
You are summarizing an entire video based on aggregated segment summaries.

INPUT:
A list of segment-level or window-level summaries.

TASK:
Produce a coherent, high-level understanding of the video.

OUTPUT STRUCTURE:

1. OVERVIEW
- A concise description of what the video is about

2. TIMELINE OF KEY EVENTS
- Ordered bullet points

3. MAIN ENTITIES
- People, objects, or recurring elements

4. IMPORTANT ACTIONS
- Core actions driving the video

5. THEMES OR PATTERNS (if any)
- Only if clearly supported by the data

RULES:
- Do NOT repeat information unnecessarily
- Do NOT hallucinate missing details
- Stay faithful to input
- Be concise but complete\
"""

# ── Stage 4: iterative refinement prompt (optional, reserved for future use) ──
# Included from the design document for streaming mode where segments are
# processed incrementally rather than batched.  Not currently wired into the
# pipeline but available for future extensions.
ITERATIVE_REFINEMENT_PROMPT = """\
You are maintaining a running summary of a video.

CURRENT SUMMARY:
{existing_summary}

NEW SEGMENT DATA:
{new_segment_json}

TASK:
Update the summary to incorporate new information.

RULES:
- Merge, don't append blindly
- Remove redundant information
- Keep the summary concise and coherent
- Preserve chronological structure

OUTPUT:
Updated summary only.\
"""

# ── Default max-new-tokens per stage ─────────────────────────────────────
# The design document recommends smaller budgets for segment-level and larger
# for aggregation / final stages.
DEFAULT_SEGMENT_MAX_TOKENS = 600
DEFAULT_WINDOW_MAX_TOKENS = 2048
DEFAULT_FINAL_MAX_TOKENS = 2048
