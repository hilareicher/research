CHECK_PROMPT_TEMPLATE = """
You are performing a step-by-step analysis of a medical record to detect overt physical aggression toward others. This task requires strict attention to identifying **actual physical violence directed at other persons**, and not any other forms of aggression or distress.

Aggression keywords to look for (only when describing **physical violence against others**): הכה, הרביץ, דחף, בעט, חבט, זרק, דקר, נשך, חנק, תקף, התנהגות תוקפנית, איים, תקיפה.

You must not repeat, summarize, paraphrase, or explain any part of the EMR.
Do not generate steps, chain-of-thought, or commentary.
Do not include any reasoning.
Do not mention the EMR or reuse any content from it, except a single quoted aggression snippet if applicable.

This is not a free-form task. Your answer must follow this **exact format** — no deviations:
```json
{
  "actual": true | false,
  "justification": "<verbatim snippet or empty>"
}
```

Instructions:
1. Carefully scan the EMR text for any of the above keywords indicating physical aggression directed at another person.
2. Confirm that the context **clearly** describes **actual physical aggression toward others** (not threats, not verbal aggression, not self-harm, not general agitation or emotional distress). If it's not unambiguous physical violence directed at another person, mark it as `false`.
3. If and ONLY if aggression toward others is explicitly confirmed:
   - Set `actual = true`
   - Extract the exact verbatim snippet containing the aggression keyword, **wrapped in double quotes**, and assign it to `justification`.
4. If there is **any doubt, ambiguity, or no aggression**:
   - Set `actual = false`
   - Set `justification = ""` (an empty string — do NOT explain why)

Any output that deviates from the required format will be considered invalid.

EMR:
{emr_text}
"""