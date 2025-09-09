SYSTEM_PROMPT_TEMPLATE = """You are {author_name}, a writer with the following characteristics:

Tone: {tone}
Voice: {voice}
Formality: {formality}
Preferred Length: {length_preference}

Style Notes: {writing_style_notes}

{topic_guidance}

Please write in a way that reflects these characteristics consistently."""

DATASET_BUILDING_PROMPTS = {
    "writing_sample": """I'd like to help you create training examples for your writing style.

Please provide a writing sample that represents your style. This could be:
- A paragraph from a blog post
- An email you've written
- A social media post
- Any other text that feels like "your voice"

Writing sample:""",
    "prompt_for_sample": """Based on this writing sample, what kind of prompt or question would lead someone to write this response?

For example:
- If it's a blog post intro, the prompt might be "Write an introduction about [topic]"
- If it's an email, the prompt might be "Draft a professional email about [subject]"
- If it's creative writing, the prompt might be "Write a short story about [theme]"

What prompt would generate this writing sample?""",
    "topic_extraction": """What is the main topic or theme of this writing sample?

Please provide a brief 1-2 word topic (e.g., "technology", "productivity", "personal story", "business update"):""",
    "style_analysis": """Based on this writing sample, how would you describe your writing style?

Consider:
- Tone (professional, casual, friendly, authoritative, etc.)
- Formality level (very casual, conversational, formal, academic)
- Length preference (concise, moderate, detailed)
- Any unique characteristics

Your style description:""",
}

EXAMPLE_GENERATION_PROMPTS = {
    "blog_intro": "Write an engaging introduction for a blog post about {topic}",
    "email_draft": "Draft a {tone} email about {topic}",
    "social_post": "Write a {tone} social media post about {topic}",
    "article_section": "Write a {length_preference} section for an article about {topic}",
    "personal_reflection": "Write a personal reflection on {topic}",
    "how_to_guide": "Write the introduction to a how-to guide about {topic}",
    "product_description": "Write a {tone} product description for {topic}",
    "newsletter_segment": "Write a newsletter segment about {topic}",
}

EXAMPLE_GENERATION_FROM_EXISTING_TEMPLATE = """Based on the following examples from a user's writing style, generate new training examples that match the same tone, voice, and format.

EXISTING EXAMPLES:
{existing_examples}

Requirements for new examples:
1. Match the writing style, tone, and voice shown above
2. Use different topics/prompts but maintain similar response patterns  
3. Keep similar length and complexity as existing examples
4. Show variety in content while preserving the author's unique style
5. Make responses natural and authentic to the established voice

Generate {count} new training example(s) with this exact format:

EXAMPLE 1:
User prompt: [Create a new prompt similar to those shown above]
Assistant response: [Write response in the established style]

EXAMPLE 2:
User prompt: [Different prompt, same style]  
Assistant response: [Response matching the author's voice]

[Continue for {count} examples...]"""

REVERSE_ENGINEER_PROMPT_TEMPLATE = """Analyze the following content and determine what prompt would most likely generate this exact response.

CONTENT TO ANALYZE:
{content}

Based on this content, create a specific, actionable prompt that would lead someone to write this response. Consider:
- The main topic and purpose
- The writing style and tone  
- The format and structure
- The intended audience
- The specific angle or approach

Provide only the prompt, nothing else. Make it clear, specific, and actionable.

PROMPT:"""


def build_system_prompt(author_profile) -> str:
    style = author_profile.style_guide

    topic_guidance = ""
    if style.topics:
        topic_guidance += f"Preferred topics: {', '.join(style.topics)}\n"
    if style.avoid_topics:
        topic_guidance += f"Topics to avoid: {', '.join(style.avoid_topics)}\n"

    return SYSTEM_PROMPT_TEMPLATE.format(
        author_name=author_profile.name,
        tone=style.tone,
        voice=style.voice,
        formality=style.formality,
        length_preference=style.length_preference,
        writing_style_notes=style.writing_style_notes or "No additional notes",
        topic_guidance=topic_guidance.strip(),
    )


def get_example_prompt(prompt_type: str, **kwargs) -> str:
    if prompt_type in EXAMPLE_GENERATION_PROMPTS:
        return EXAMPLE_GENERATION_PROMPTS[prompt_type].format(**kwargs)
    return f"Write about {kwargs.get('topic', 'the given topic')}"
