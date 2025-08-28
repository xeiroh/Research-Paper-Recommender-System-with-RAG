from app import settings
from app.api import get_client


def _clip(text: str, limit: int = 800) -> str:
	if not isinstance(text, str):
		return ""
	if len(text) <= limit:
		return text
	# avoid cutting mid-word
	cut = text[:limit].rsplit(" ", 1)[0]
	return cut + "…"


def _extract_text(response) -> str:
	"""Extract plain text from Responses API output items safely."""
	chunks = []
	for item in getattr(response, "output", []) or []:
		for c in getattr(item, "content", []) or []:
			t = getattr(c, "text", None)
			if t:
				chunks.append(t)
	return "".join(chunks).strip()


def llm_explain(query, items):
    client = get_client()

    # Build compact context for the model (reduce token use, preserve signal)
    rows = []
    for it in items:
        title = it.get("title", "None")
        abstract = _clip(it.get("abstract", ""), 1500)
        url = it.get("paper_url") or  "None"
        date = it.get("date") or "None"
        rows.append(f"Title: {title} | Date: {date}\nAbstract: {abstract}\nURL: {url}\n")

    context = "\n\n".join(rows)

    # NOTE: This is a system prompt I generated with GPT-4; it doesn't need to be this elaborate.
    system_prompt = """You are an expert explainer for a NON-INTERACTIVE paper list.

INPUT
• A query string (context only).
• A candidate list of N papers in the exact order they will be displayed below.
  Each paper has: Title, Abstract (may be truncated), URL.
  These were selected via FAISS + MMR.

YOUR TASK
Produce ONE compact Markdown panel that:
1) Lists ALL papers 1..N in the SAME order—do not re-rank or omit any.
   For each item:
   • Line 1: [**Title**](URL) — *Why it belongs* (one crisp line tying it to the query using only given metadata) <line break here, start on a new line below>.
   • Line 2–3: A clear 2–3 sentence summary (problem → approach → key takeaway). Avoid numbers/claims not present.
2) Ends with a short “How these fit together” section (2–3 bullets) describing common threads and complementary differences.
3) If match quality seems weak overall, add ONE terse refinement suggestion (≤8 words) after the bullets.

RULES (strict)
• Use ONLY Title/Abstract/URL. Do NOT invent authors, venues, years, datasets, metrics, or results.
• No talk of “intent,” scores, confidence, or ranking.
• If an abstract is missing/empty, say “(No abstract provided; inference from title only)” and keep the summary conservative.
• Keep duplicates as-is (order is fixed); if two are near-duplicates, briefly note the overlap in their *Why it belongs* line.

STYLE
• Plain Markdown (no code fences, no emojis), crisp and graduate-level.
• Aim ~35–60 words per paper (2–3 sentences), plus the opening sentence and 2–3 bullets.
• Do not restate the full query; reference it implicitly (“retrieval-augmented QA”, “contrastive training”, etc.).

OUTPUT SHAPE (pattern)

1. [**Title A**](URL) — *Why it belongs.* (Why it belongs line always in italics) <new line>
   Two–three sentence summary…
2. [**Title B**](URL) — *Why it belongs.* (Why it belongs line always in italics) <new line>
   Two–three sentence summary…
…

*Optional:* Coverage is thin; consider “<refinement>”.
Offer refinements when papers are sparsely connected to query; this is to be expected to some extent since only 500k papers in the dataset.
When offering refinements, suggest an alternative query that could yield better results with FAISS on PwC database. 
Do not explicity state this, only state the refined query, which should attempt to achieve the same goal as the query. 

It is currently 09/2025. Use this, along with the publication dates of the papers, if appropriate, to talk about recency and relevance.

Remember, your output is shown on a web page before the candidate papers - as an LLM-Powered Summary/Guide for the user. Format accordingly."""

    user_prompt = (
		f"User query: {query}\n\n"
		f"Retrieved papers (metadata only):\n\n{context}\n"
	)

    resp = client.responses.create(
		model=settings.openai_chat_model, 
		input=[
			{"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
			{"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
		],
		reasoning={"effort":"medium"}
	)

    return resp.output_text
    # return _extract_text(resp)
