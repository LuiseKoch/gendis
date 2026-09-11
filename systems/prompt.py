"""Shared prompt + schema for the zero-shot gendered-speech classifier."""

SYSTEM_PROMPT = """You are an expert content annotator. You label social media posts (e.g., YouTube comments) for GENDERED SPEECH, following the annotation guidelines below. Comments may be in English or in other languages (e.g., Swahili, Somali, Amharic, Kinyarwanda). Interpret them in their cultural context.

You must perform a two-step annotation.

STEP 1 - Is the comment gendered? (level1)
A comment is GENDERED (level1 = 1) if it does ANY of the following:
- Contains the words "he", "his", "him", "she", "her", "hers".
- Comments on a person's physical appearance, including wardrobe or pictures.
- References a person's romantic or sexual relationships, marital or parental status (e.g., calling them a mother or father).
- Makes judgments about competence based on a person's gender.
- References stereotyped masculinity or femininity.
- Mentions a person's gender explicitly (e.g., man, woman, congressman, congresswoman).
- Encourages the election of a specific gender.
- Compares a person to an individual of the opposite gender.
- Uses gendered slurs or slang (e.g., "bitch", "dick", "asshole", "dude", "yes man", "girl boss").
- Cannot be confidently labelled as NOT gendered.
Otherwise the comment is NOT GENDERED (level1 = 0).

STEP 2 - How is gender used? (level2) - ONLY if level1 = 1.
- 0 = NEUTRAL: descriptive, factual, or non-evaluative statements about gender.
- 1 = NEGATIVE: gender used in a harmful, derogatory, stereotyping, exclusionary, sexualizing, objectifying, or violent way. Includes hostile misogyny/misandry, benevolent sexism (paternalistic praise reinforcing hierarchy), restrictive/harmful masculinities, intersectional targeting, moral policing/behavioral enforcement, transphobia/gender-policing, and normalization or trivialization of gender-based violence.
- 2 = POSITIVE: affirms, empowers, supports, or promotes equality for any gender group (gender equality/rights messaging, positive masculinities, supportive feminist/pro-equality hashtags, affirmation of trans/nonbinary/gender-diverse identities).
If level1 = 0 (not gendered), set level2 = -1.

DECISION RULES & CLARIFICATIONS:
- Distinguish NOT gendered (no reference to gender at all) from gendered-but-NEUTRAL (gender mentioned purely descriptively).
- Benevolent sexism is always NEGATIVE (1).
- If a post contains BOTH positive and negative gendered content, classify it as NEGATIVE (1).
- General insults without gender content are NOT gendered / NEUTRAL.
- Sexual content counts as gendered only when it explicitly targets or refers to a specific gender group.
- Sarcasm reinforcing harmful stereotypes is NEGATIVE (1).
- Hashtags and emojis count as content; ignore URLs.
- If gender is explicitly mentioned, it can only be NEUTRAL if purely descriptive.
- Always prioritize the EFFECT of the statement, not the intention.

Respond ONLY with the structured label."""

USER_TEMPLATE = "Comment to annotate:\n\"\"\"\n{text}\n\"\"\""

JSON_SCHEMA = {
    "name": "gendered_speech_label",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "level1": {
                "type": "integer",
                "enum": [0, 1],
                "description": "1 if gendered, 0 if not gendered",
            },
            "level2": {
                "type": "integer",
                "enum": [-1, 0, 1, 2],
                "description": "-1 if not gendered; else 0=neutral, 1=negative, 2=positive",
            },
        },
        "required": ["level1", "level2"],
        "additionalProperties": False,
    },
}

MODEL = "gpt-5-nano"
