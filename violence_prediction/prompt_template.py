CHECK_PROMPT_TEMPLATE = """
You are performing a step-by-step analysis of a medical record to detect overt physical aggression toward others.

Aggression keywords to look for: הכה, הרביץ, דחף, בעט, חבט, זרק, דקר, נשך, חנק, תקף, התנהגות תוקפנית, איים, תקיפה.

**Chain of Thought**:
1. Carefully scan the EMR text for any of the above keywords indicating physical aggression directed at another person.
2. Confirm that the context specifically describes aggression toward others (not self-harm or non-violent behavior).
3. If and only if aggression is clearly present:
   - set `actual = yes`
   - extract exactly the verbatim snippet containing the keyword, wrapped in double quotes, and assign it to `justification`.
4. If there is no aggression present:
   - set `actual = no`
   - set `justification = ""` (do NOT supply any snippet).

Very important:
- Your response must strictly follow the exact format below. No extra commentary or explanation is allowed.
- You must ALWAYS output a Python code block, exactly as shown below.
- If there is any uncertainty, default to `actual = no`.

Respond ONLY with the following Python assignments, and nothing else:
```python
actual = <yes/no>
justification = "<verbatim snippet or empty>"
```

EMR:
{emr_text}
"""