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