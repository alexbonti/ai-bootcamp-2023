from googletrans import Translator
import textwrap

def translate_large_text(text, translate_function, choice, max_length=500):
    """
    Break down large text, translate each part, and merge the results.

    :param text: str, The large body of text to translate.
    :param translate_function: function, The translation function to use.
    :param max_length: int, The maximum character length each split of text should have.
    :return: str, The translated text.
    """
    
    # Split the text into parts of maximum allowed character length.
    text_parts = textwrap.wrap(text, max_length, break_long_words=True, replace_whitespace=False)

    translated_text_parts = []

    for part in text_parts:
        # Translate each part of the text.
        translated_part = translate_function(part, choice)  # Assuming 'False' is a necessary argument in the actual function.
        translated_text_parts.append(translated_part)

    # Combine the translated parts.
    full_translated_text = ' '.join(translated_text_parts)

    return full_translated_text


def translate_to_thai(sentence: str, choice: bool) -> str:
    """
    Translate the text between English and Thai based on the 'choice' flag.
    
    Args:
        sentence (str): The text to translate.
        choice (bool): If True, translates text to Thai. If False, translates to English.

    Returns:
        str: The translated text.
    """
    translator = Translator()
    try:
        if choice:
            # Translate to Thai
            translate = translator.translate(sentence, dest='th')
        else:
            # Translate to English
            translate = translator.translate(sentence, dest='en')
        return translate.text
    except Exception as e:
        # Handle translation-related issues (e.g., network error, unexpected API response)
        raise ValueError(f"Translation failed: {str(e)}") from e