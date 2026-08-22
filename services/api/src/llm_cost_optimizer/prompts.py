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
  "points": [
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
  "verdict": "<GOOD or BAD>"
}}
"""

EXTRACTION_JUDGE_PROMPT_TEMPLATE = """
You are an exacting evaluator that checks whether an EXTRACTION is faithful to its source.
Do NOT perform the extraction yourself and do NOT correct it — only evaluate what you are given.

Inputs:
- PROMPT: The extraction instructions plus the source text the values were supposed to come from.
- EXTRACTED: The structured output a model produced from that source.

Tasks (follow exactly):
1) Ground every extracted value. For each field/value present in EXTRACTED, decide:
   - SUPPORTED    : the value appears in, or follows directly from, the source text in PROMPT.
   - UNSUPPORTED  : the source does not contain enough information to confirm the value, even if
                    it sounds plausible. A value invented to fill a required field is UNSUPPORTED.
   - CONTRADICTED : the source states something different from the extracted value.
2) Missing items: list any field the PROMPT asked for whose value IS present in the source text but
   is absent, empty, or null in EXTRACTED. Do NOT list fields that are genuinely absent from the
   source — correctly reporting nothing is not a miss. May be an empty array.
3) Format adherence: a number between 0.0 and 1.0 for how well EXTRACTED follows the output shape,
   schema, field names, and types the PROMPT requested (higher is better). If the PROMPT specifies
   no particular format, score 1.0.

Judge substance only. Do not reward verbosity, confident tone, or extra fields nobody asked for.

PROMPT:
{prompt}

EXTRACTED:
{extracted}
"""


QA_JUDGE_PROMPT_TEMPLATE = """
You are an exacting evaluator that checks whether an ANSWER correctly answers the question
asked in a PROMPT. Do NOT answer the question yourself — only evaluate the answer given.

Inputs:
- PROMPT: The question, plus any context the asker supplied.
- ANSWER: The answer produced by a model.

Grounding rule for this evaluation:
{grounding_rule}

Tasks (follow exactly):
1) Correctness: decide whether ANSWER is CORRECT, PARTIALLY_CORRECT, or INCORRECT.
2) Unsupported claims: list any statement in ANSWER that the grounding rule above does not
   permit — a fact asserted without support (may be an empty array).
3) Missing aspects: list any part of the question that ANSWER leaves unaddressed
   (may be an empty array).
4) Directness: a number between 0.0 and 1.0 for how directly ANSWER responds to the question
   actually asked. An answer that is true but evasive, padded, or answers a different question
   scores low here even when correctness is high.

Judge substance only. Length, confident tone, and formatting polish must not affect any score.

PROMPT:
{prompt}

ANSWER:
{answer}
"""


GENERATION_JUDGE_PROMPT_TEMPLATE = """
You are an exacting evaluator of generated text. Do NOT rewrite or improve the text —
only evaluate it against what the PROMPT asked for.

Inputs:
- PROMPT: The generation request, including any constraints (length, format, audience, tone).
- GENERATED: The text a model produced.

Tasks (follow exactly):
1) Instruction adherence: a number between 0.0 and 1.0 for how completely GENERATED does what
   the PROMPT asked.
2) Coherence: a number between 0.0 and 1.0 for internal consistency and logical flow. Penalize
   self-contradiction, abrupt topic drift, and repetition.
3) Relevance: a number between 0.0 and 1.0 for how well GENERATED stays on the requested subject.
4) Violated constraints: list any explicit constraint in the PROMPT that GENERATED breaks —
   wrong length, wrong format, forbidden content, wrong point of view (may be an empty array).
5) Unsupported claims: list any factual assertion presented as true that the PROMPT does not
   support and that a careful reader would want checked (may be an empty array).

Do not reward length or confident tone. Judge substance only.

PROMPT:
{prompt}

GENERATED:
{generated}
"""


CLASSIFICATION_JUDGE_PROMPT_TEMPLATE = """
You are an exacting evaluator of a classification decision. Do NOT reclassify the item
yourself as your primary task — only evaluate the label that was produced.

Inputs:
- PROMPT: The classification instructions, the label set (if one was given), and the item.
- LABEL: The classification output a model produced.

Tasks (follow exactly):
1) Label valid: true only if LABEL is one of the labels the PROMPT allows. If the PROMPT
   specifies no fixed label set, judge whether LABEL is a sensible category for the task and
   set this true. Inventing a label outside a stated set is always false.
2) Correctness: decide whether LABEL is CORRECT, AMBIGUOUS, or INCORRECT for the item.
   Use AMBIGUOUS only when the item genuinely supports more than one label and LABEL is one
   of the defensible choices.
3) Format adherence: a number between 0.0 and 1.0 for how well the output matches the shape
   the PROMPT requested (bare label vs JSON, casing, extra commentary). If the PROMPT requested
   no particular format, score 1.0.
4) Reasoning: one or two sentences justifying your correctness decision, citing the item.

PROMPT:
{prompt}

LABEL:
{label}
"""


CODE_GENERATION_JUDGE_PROMPT_TEMPLATE = """
You are an exacting code reviewer evaluating generated code. Do NOT rewrite the code —
only evaluate it. Assume the code will be run as-is.

Inputs:
- PROMPT: The coding request, including language, constraints, and any spec or signature.
- CODE: The code a model produced.

Tasks (follow exactly):
1) Requirements met: decide whether CODE satisfies what the PROMPT asked: MET,
   PARTIALLY_MET, or NOT_MET.
2) Syntax valid: true if CODE would parse in the target language. Judge syntax only here,
   not behavior. Truncated or obviously incomplete code is false.
3) Issues: list concrete defects. For each, give a severity and a specific description that
   references the actual code:
   - CRITICAL : will not run, or is wrong for the primary case the PROMPT asked about;
                also security holes such as injection or leaked secrets.
   - MAJOR    : wrong on an important edge case, a real resource leak, or a missing
                requirement the PROMPT stated explicitly.
   - MINOR    : style, naming, a small inefficiency, a missing docstring.
   May be an empty array. Do not invent issues to seem thorough.
4) Format adherence: a number between 0.0 and 1.0 for whether CODE is in the requested
   language and structure (a bare function vs a full module vs a diff), and whether it
   respects a signature the PROMPT specified. If the PROMPT requested no particular
   format, score 1.0.

PROMPT:
{prompt}

CODE:
{code}
"""


GENERIC_JUDGE_PROMPT_TEMPLATE = """
You are an exacting evaluator of an assistant's reply. Do NOT answer the user yourself —
only evaluate the reply given.

Inputs:
- PROMPT: What the user asked for.
- REPLY: The reply an assistant produced.

Tasks (follow exactly):
1) Helpfulness: a number between 0.0 and 1.0 for how much the REPLY actually advances the
   user's goal. A reply that refuses a reasonable request, or answers a question the user
   did not ask, scores low.
2) Correctness: a number between 0.0 and 1.0 for factual and logical accuracy of what REPLY
   asserts. If REPLY makes no checkable claims, score 1.0.
3) Instruction adherence: a number between 0.0 and 1.0 for how well REPLY respects explicit
   instructions in the PROMPT — format, length, tone, things to avoid.
4) Unsupported claims: list any factual assertion in REPLY that is presented as true but is
   not supported by the PROMPT and that a careful reader would want checked
   (may be an empty array).

Do not reward length, confident tone, or formatting polish. Judge substance only.

PROMPT:
{prompt}

REPLY:
{reply}
"""
