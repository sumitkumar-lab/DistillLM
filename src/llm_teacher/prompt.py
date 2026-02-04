INTENTS = [
    "LIGHT_ON",
    "LIGHT_OFF",
    "AC_ON",
    "AC_OFF",
    "PLAY_MUSIC",
    "STOP_MUSIC"
]

def build_prompt(text):
    return f"""
You are an intent classification system for an embedded device.

Return a JSON object with intent probabilities (sum to 1).
No explanation.

Intents:
{', '.join(INTENTS)}

User command:
"{text}"
"""
