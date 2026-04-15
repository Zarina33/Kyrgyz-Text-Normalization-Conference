"""
Rule-based baseline for Kyrgyz text correction.
Applies simple capitalization and punctuation rules.
"""

import re


def correct(text: str) -> str:
    if not text or not text.strip():
        return text

    text = text.strip()

    # Capitalize first letter
    text = text[0].upper() + text[1:]

    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)

    # Add period at end if no sentence-ending punctuation
    if text and text[-1] not in '.!?…':
        text = text + '.'

    # Capitalize after sentence-ending punctuation
    def capitalize_after_punct(m):
        return m.group(1) + ' ' + m.group(2).upper()

    text = re.sub(r'([.!?])\s+([а-яёa-zа-яөүңкгжшчщ])', capitalize_after_punct, text)

    return text


if __name__ == "__main__":
    examples = [
        "мен бүгүн дүкөнгө бардым",
        "ШАИР БИР ЕРКЕКЧИЛИК КЫЛЫБ ОЛКОНУН КАРЫЗЫН ТОЛОБ КОЙЧУ",
        "ал китеп окуду биз мектепте окуйбуз",
    ]
    for ex in examples:
        print(f"IN:  {ex}")
        print(f"OUT: {correct(ex)}")
        print()
