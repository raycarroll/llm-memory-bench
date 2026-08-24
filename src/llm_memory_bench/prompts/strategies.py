from pathlib import Path

MINIMAL = (
    "You are a helpful assistant with access to a memory tool. "
    "Use it to save important facts about the user for future conversations."
)

DETAILED = (
    "You are a helpful assistant with access to a memory tool that persists "
    "information across conversations.\n\n"
    "SAVE using the add_memory tool when the user shares:\n"
    "- Personal preferences (likes, dislikes, styles)\n"
    "- Background information (job, education, expertise, location)\n"
    "- Constraints or requirements (allergies, accessibility needs)\n"
    "- Long-term goals or ongoing projects\n"
    "- Technical environment (languages, tools, OS)\n\n"
    "DO NOT save:\n"
    "- Greetings or small talk\n"
    "- One-off task requests ('fix this bug', 'translate this sentence')\n"
    "- Information the user is asking about, not stating about themselves\n"
    "- Ephemeral context ('I'm in a hurry', 'just a quick question')\n"
    "- Facts about other people unless the user explicitly asks you to remember them\n\n"
    "When in doubt, do not save. Only store facts you'd want available in a "
    "completely new conversation with this user."
)

FEW_SHOT = (
    "You are a helpful assistant with access to a memory tool. "
    "Use it to save important, persistent facts about the user.\n\n"
    "Examples of CORRECT usage:\n\n"
    'User: "I\'m a data scientist working mostly in Python and R."\n'
    "→ Call add_memory with: \"User is a data scientist, works primarily in Python and R\"\n\n"
    'User: "Please never use ORMs in your code suggestions, I prefer raw SQL."\n'
    "→ Call add_memory with: \"User prefers raw SQL over ORMs in code suggestions\"\n\n"
    "Examples of when NOT to use the tool:\n\n"
    'User: "Can you help me debug this function?"\n'
    "→ Do NOT call add_memory (this is a one-off task request)\n\n"
    'User: "Good morning! How are you?"\n'
    "→ Do NOT call add_memory (this is small talk)\n\n"
    'User: "What\'s the capital of France?"\n'
    "→ Do NOT call add_memory (user is asking a question, not sharing personal info)"
)

STRATEGIES = {
    "minimal": MINIMAL,
    "detailed": DETAILED,
    "few_shot": FEW_SHOT,
}


def get_system_prompt(strategy: str) -> str:
    if strategy in STRATEGIES:
        return STRATEGIES[strategy]

    path = Path(strategy)
    if path.is_file():
        return path.read_text().strip()

    raise ValueError(
        f"Unknown strategy: {strategy}. "
        f"Available: {list(STRATEGIES.keys())} or provide a file path."
    )
