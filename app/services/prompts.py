STATE_PROMPT = """You maintain the structured diagnosis state of a car-repair conversation.
Given the conversation and the previous state (JSON), output ONLY a JSON object, with no prose
and no code fences, using exactly these keys:
{
  "suspected_causes": [{"cause": str, "likelihood": number between 0 and 1, "evidence": str, "next_check": str}],
  "safety_flags": [str],
  "title": str (at most 60 characters)
}
Update the previous state with anything new from the conversation: revise likelihoods, add or
drop causes, and list safety concerns (for example "do not keep driving while overheating").
Keep the language of the conversation."""
