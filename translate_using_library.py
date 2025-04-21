from googletrans import Translator as GoogleTranslatorLib
from deep_translator import GoogleTranslator as DeepGoogleTranslator
import argostranslate.translate 
import time

def translate_googletrans(text, src, dest):
    translator = GoogleTranslatorLib()
    try:
        start = time.time()
        result = translator.translate(text, src=src, dest=dest)
        return {"translation": result.text, "inference_time": time.time() - start}
    except Exception as e:
        return {"translation": f"googletrans error: {e}", "inference_time": None}

def translate_deeptranslator(text, src, dest):
    try:
        start = time.time()
        result = DeepGoogleTranslator(source=src, target=dest).translate(text)
        return {"translation": result, "inference_time": time.time() - start}
    except Exception as e:
        return {"translation": f"deep-translator error: {e}", "inference_time": None}

def translate_argos(text, src, dest):
    try:
        start = time.time()
        result = argostranslate.translate.translate(text, from_code=src, to_code=dest)
        return {"translation": result, "inference_time": time.time() - start}
    except Exception as e:
        return {"translation": f"argos-translate error: {e}", "inference_time": None}

def main():
    print(" English ⇄ Dari Translator Test With Library")
    language = input("Translate from (en/fa): ").strip().lower()
    if language not in ("en", "fa"):
        print("Invalid direction! Use 'en' or 'fa'.")
        return

    target = "fa" if language == "en" else "en"
    text = input(f"Enter text in {language}: ").strip()

    print("\n--- Translations ---")
    print(f"googletrans:      {translate_googletrans(text, src=language, dest=target)}")
    print(f"deep-translator:  {translate_deeptranslator(text, src=language, dest=target)}")
    print(f"argos-translate:  {translate_argos(text, src=language, dest=target)}")

if __name__ == "__main__":
    main()
