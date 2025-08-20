CHECK_PROMPT_TEMPLATE = """
[SYSTEM: You are an API that MUST return ONLY valid JSON. No other text, explanations, or commentary is allowed]

Below is a medical record that needs to be analyzed for overt physical aggression toward others. Your task is to return a JSON object indicating whether there is clear evidence of physical violence directed at another person, following the EXACT structure specified.

REQUIRED OUTPUT FORMAT:
{
  "actual": boolean,    // must be exactly true or false, no quotes
  "justification": ""   // must be exactly "", or a quoted string if violence found
}

STRICT RULES:
1. You MUST output ONLY the JSON object above
2. No explanations, no commentary, no other text
3. No HTML, no markdown, no code blocks
4. The "actual" field MUST be a boolean (true/false), not a string
5. The "justification" field MUST be either an empty string or a quoted snippet
6. Do not repeat or explain the EMR content
7. ANY deviation from this format will cause rejection

Keywords indicating physical aggression (only when describing actual violence against others):
הכה, הרביץ, דחף, בעט, חבט, זרק, דקר, נשך, חנק, תקף, התנהגות תוקפנית, איים, תקיפה

DECISION LOGIC:
- Set actual=true ONLY if there is unambiguous physical violence against others
- Set actual=false for threats, verbal aggression, self-harm, or general agitation
- When actual=true, justification must contain the exact violence snippet
- When actual=false, justification must be an empty string

EMR TEXT:
{emr_text}

[SYSTEM: Remember - return ONLY the JSON object. No other output is allowed.]
"""