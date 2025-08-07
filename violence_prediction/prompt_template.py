CHECK_PROMPT_TEMPLATE = """
You are performing a step-by-step analysis of a medical record to detect overt physical aggression toward others.

Aggression keywords to look for: הכה, הרביץ, דחף, בעט, חבט, זרק, דקר, נשך, חנק, תקף, התנהגות תוקפנית, איים, תקיפה.

**Chain of Thought**:
1. Scan the EMR text for any of the above keywords indicating physical aggression directed at another person.
2. Confirm the context describes aggression toward others (not self-harm or non-violent behavior).
3. If aggression is present:
   - set `actual = yes`
   - extract exactly the verbatim snippet containing the keyword, wrapped in double quotes, and assign it to `justification`.
4. If no aggression is found:
   - set `actual = no`
   - set `justification = ""` (do NOT supply any snippet).

Answer only with the following Python assignments, with no additional commentary:
```python
actual = <yes/no>
justification = "<verbatim snippet or empty>"
```

EMR:
{emr_text}
"""