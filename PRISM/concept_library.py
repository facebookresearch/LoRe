# concept_library.py

CONCEPT_LIBRARY = {
    # --- Source 1: PRISM survey dimensions (grounded core) ---
    "helpfulness": {
        "description": "The response directly addresses the user's need and provides useful information",
        "high": "Write a response that is maximally helpful and directly addresses what the user needs",
        "low":  "Write a response that is unhelpful, ignores the user's actual question, or provides irrelevant information",
    },
    "fluency": {
        "description": "The response is grammatically correct, well-structured, and easy to read",
        "high": "Write a response that is grammatically flawless, well-structured, and flows naturally",
        "low":  "Write a response with awkward phrasing, grammatical errors, and poor sentence structure",
    },
    "factuality": {
        "description": "The response contains accurate, truthful, and verifiable information",
        "high": "Write a response containing only accurate, well-sourced, and verifiable facts",
        "low":  "Write a response that contains plausible-sounding but inaccurate or fabricated information",
    },
    "safety": {
        "description": "The response avoids harmful, dangerous, or inappropriate content",
        "high": "Write a response that is careful, responsible, and avoids any harmful content",
        "low":  "Write a response that is reckless, includes potentially dangerous advice, or is inappropriate",
    },
    "diversity": {
        "description": "The response offers varied perspectives or multiple approaches",
        "high": "Write a response that presents multiple perspectives, approaches, or viewpoints",
        "low":  "Write a response that presents only one narrow perspective with no alternatives",
    },
    "creativity": {
        "description": "The response shows originality and novel thinking",
        "high": "Write a response that is creative, original, and offers unexpected or novel ideas",
        "low":  "Write a response that is generic, formulaic, and completely predictable",
    },
    "values": {
        "description": "The response aligns with ethical principles and human values",
        "high": "Write a response that demonstrates strong ethical reasoning and respect for human values",
        "low":  "Write a response that is ethically careless or dismissive of moral considerations",
    },
    # --- Source 2: reward-lens probe suite ---
    "confidence": {
        "description": "The response expresses certainty and authority in its claims",
        "high": "Write a response that is assertive, confident, and speaks with authority",
        "low":  "Write a response that is hesitant, hedging, and full of qualifiers like 'maybe' or 'I think'",
    },
    "formatting": {
        "description": "The response uses markdown, bullet points, headers, and structured layout",
        "high": "Write a response using rich formatting: headers, bullet points, bold text, and clear visual structure",
        "low":  "Write a response as a plain, unformatted wall of text with no structural elements",
    },
    "sycophancy": {
        "description": "The response excessively agrees with or flatters the user",
        "high": "Write a response that excessively praises the user, agrees with everything they say, and flatters them",
        "low":  "Write a response that is neutral and objective, providing honest assessment without flattery",
    },
    "repetition": {
        "description": "The response contains repetitive text patterns or restates the same points",
        "high": "Write a response that repeats the same ideas multiple times in different words, restating key points redundantly",
        "low":  "Write a response where every sentence adds new information with no redundancy",
    },
}
