SUMMARIZATION_JUDGE_PROMPT_TEMPLATE = """You are a strict fact-checking judge. You will be given a SOURCE DOCUMENT and a SUMMARY produced by another model. Your job is NOT to rewrite or improve the summary. Your job is only to check it.
 
Step 1 - Extract claims: List every distinct factual claim made in the SUMMARY (short phrases, one per line).
 
Step 2 - Verify claims: For each claim, decide if it is SUPPORTED, CONTRADICTED, or UNSUPPORTED by the SOURCE DOCUMENT. A claim is UNSUPPORTED if the source does not contain enough information to confirm it, even if it sounds plausible.
 
Step 3 - Extract key points: List the 3 to 5 most important points a reader would need from the SOURCE DOCUMENT.
 
Step 4 - Check coverage: For each key point, decide if it is PRESENT or MISSING in the SUMMARY.
 
Respond with ONLY valid JSON, no markdown fences, no preamble, in exactly this shape:
{{
  "claims": [
    {{"claim": "...", "status": "SUPPORTED"}},
    {{"claim": "...", "status": "UNSUPPORTED"}}
  ],
  "key_points": [
    {{"point": "...", "status": "PRESENT"}},
    {{"point": "...", "status": "MISSING"}}
  ]
}}
 
SOURCE DOCUMENT:
{source_document}
 
SUMMARY:
{summary}
"""

BRAINSTORMING_JUDGE_PROMPT_TEMPLATE = """\
SYSTEM:
You are an expert evaluator of brainstorming output. You are not the author
of these ideas and have no attachment to them. Your job is to judge quality
strictly according to the rubric below — not according to your own taste or
what you personally would have brainstormed.
 
TASK CONTEXT:
The following brainstorm was generated in response to this prompt:
<original_prompt>{original_prompt}</original_prompt>
 
Constraints the ideas were supposed to respect (if any):
<constraints>{constraints}</constraints>
 
CANDIDATE OUTPUT TO EVALUATE:
<candidate>{brainstorm_output}</candidate>
 
RUBRIC — score each dimension 1-5. Anchors below define what each score means;
do not invent your own scale.
 
1. RELEVANCE — does each idea actually address the stated prompt/constraints?
   1 = most ideas are off-topic or ignore stated constraints
   3 = ideas are on-topic but some drift or loosely satisfy constraints
   5 = every idea is clearly responsive to the prompt and constraints
 
2. NOVELTY — how much do the ideas go beyond the first, most obvious answers?
   1 = entirely generic / first-thing-that-comes-to-mind ideas
   3 = mix of obvious and some genuinely fresh angles
   5 = consistently non-obvious, shows real conceptual range
 
3. DIVERSITY — how different are the ideas from EACH OTHER (not from some
   external standard)? Penalize a list where multiple ideas are restatements
   of the same underlying concept.
   1 = ideas cluster around 1-2 underlying concepts
   3 = some clusters, some genuinely distinct directions
   5 = ideas span clearly distinct approaches/dimensions of the problem
 
4. FEASIBILITY — could a reasonably resourced team plausibly act on these?
   1 = ideas are vague, magical, or require unstated breakthroughs
   3 = plausible but under-specified (missing the "how")
   5 = concrete enough that a next step is obvious
 
5. USEFULNESS OF FRAMING — does the response help the user think, e.g. by
   grouping ideas, flagging trade-offs, or surfacing a non-obvious angle they
   didn't ask for but would want?
   1 = flat list, no synthesis
   5 = organized in a way that adds insight beyond the raw list
 
INSTRUCTIONS:
- First, in <analysis> tags, go idea-by-idea (or cluster-by-cluster) and note
  strengths/weaknesses relevant to the rubric. Be specific — reference actual
  ideas from the candidate, don't generalize.
- Do not let response length, formatting polish, or confident tone influence
  your score. Judge substance only.
- Then output ONLY a JSON object (no prose after it) in this exact schema:
 
{{
  "relevance": {{"score": <1-5>, "justification": "<1-2 sentences>"}},
  "novelty": {{"score": <1-5>, "justification": "<1-2 sentences>"}},
  "diversity": {{"score": <1-5>, "justification": "<1-2 sentences>"}},
  "feasibility": {{"score": <1-5>, "justification": "<1-2 sentences>"}},
  "framing": {{"score": <1-5>, "justification": "<1-2 sentences>"}},
  "weak_ideas_to_cut": ["<idea text or index>", ...],
  "overall_score": <1-5, your holistic judgment, need not be the average>,
  "one_line_verdict": "<single sentence summary>"
}}
"""