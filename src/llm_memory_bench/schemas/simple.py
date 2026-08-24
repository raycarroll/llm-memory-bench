SIMPLE_MEMORY_TOOL = {
    "name": "add_memory",
    "description": (
        "Store a long-term fact about the user for future sessions. "
        "Only store persistent personal information, preferences, or constraints. "
        "Do not store ephemeral requests, greetings, or task-specific details."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "description": "The fact to remember about the user.",
            },
            "category": {
                "type": "string",
                "enum": ["preference", "personal", "technical", "contextual"],
                "description": "Category of the memory.",
            },
        },
        "required": ["fact"],
    },
}


def get_tool_schema(name: str = "simple") -> dict:
    schemas = {
        "simple": SIMPLE_MEMORY_TOOL,
    }
    if name not in schemas:
        raise ValueError(f"Unknown schema: {name}. Available: {list(schemas.keys())}")
    return schemas[name]
