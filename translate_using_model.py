# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# from googletrans import Translator
# import time

# tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# translator = Translator()

# def is_english(text):
#     detected_lang = translator.detect(text).lang
#     return detected_lang == "en"

# def is_dari(text):
#     detected_lang = translator.detect(text).lang
#     return detected_lang == "fa"

# # ENGLISH TO DARI
# def translate_en_to_fa(text: str):
#     if not text.strip():
#         return {"error": "Input text cannot be empty"}
#     if not is_english(text):
#         return {"error": "Input text must be in English only"}
    
#     start_time = time.time()
#     inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
#     inputs["forced_bos_token_id"] = tokenizer.lang_code_to_id["fa_IR"]
#     outputs = model.generate(**inputs)
#     translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     inference_time = time.time() - start_time

#     return {
#         "translation": translation,
#         "inference_time": inference_time
#     }

# # DARI TO ENGLISH
# def translate_fa_to_en(text: str):
#     if not text.strip():
#         return {"error": "Input text cannot be empty"}
#     if not is_dari(text):
#         return {"error": "Input text must be in Dari/Persian only"}

#     start_time = time.time()
#     tokenizer.src_lang = "fa_IR"
#     inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
#     inputs["forced_bos_token_id"] = tokenizer.lang_code_to_id["en_XX"]
#     outputs = model.generate(**inputs)
#     translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     inference_time = time.time() - start_time

#     return {
#         "translation": translation,
#         "inference_time": inference_time
#     }

# def main():
#     print("English ⇄ Dari Translator Test With Model")
#     language = input("Translate from (en/fa): ").strip().lower()
#     if language not in ("en", "fa"):
#         print("Invalid direction! Use 'en' or 'fa'.")
#         return

#     text = input(f"Enter text in {language}: ").strip()

#     print("\n--- Translations ---")
#     if language == "en":
#         print(f"mbart-large: {translate_en_to_fa(text)}")
#     else:
#         print(f"mbart-large: {translate_fa_to_en(text)}")

# if __name__ == "__main__":
#     main()

# --------------------------------------------------------------------------------------------------

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from googletrans import Translator
import time
import streamlit as st

# MBART Model
mbart_tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", use_auth_token=st.secrets["HF_TOKEN"]")
mbart_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", use_auth_token=st.secrets["HF_TOKEN"]")

# MT5
mt5_en_fa_tokenizer = AutoTokenizer.from_pretrained("persiannlp/mt5-large-parsinlu-translation_en_fa", use_auth_token=st.secrets["HF_TOKEN"]")
mt5_en_fa_model = AutoModelForSeq2SeqLM.from_pretrained("persiannlp/mt5-large-parsinlu-translation_en_fa", use_auth_token=st.secrets["HF_TOKEN"]")

mt5_fa_en_tokenizer = AutoTokenizer.from_pretrained("persiannlp/mt5-base-parsinlu-opus-translation_fa_en", use_auth_token=st.secrets["HF_TOKEN"]")
mt5_fa_en_model = AutoModelForSeq2SeqLM.from_pretrained("persiannlp/mt5-base-parsinlu-opus-translation_fa_en", use_auth_token=st.secrets["HF_TOKEN"]")

translator = Translator()

def is_english(text):
    return translator.detect(text).lang == "en"

def is_dari(text):
    return translator.detect(text).lang == "fa"

# MBART
def translate_en_to_fa(text):
    if not text.strip():
        return {"error": "Input text cannot be empty"}
    if not is_english(text):
        return {"error": "Input text must be in English only"}
    
    mbart_tokenizer.src_lang = "en_XX"
    inputs = mbart_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    inputs["forced_bos_token_id"] = mbart_tokenizer.lang_code_to_id["fa_IR"]
    start = time.time()
    output = mbart_model.generate(**inputs)
    translation = mbart_tokenizer.decode(output[0], skip_special_tokens=True)
    return {"translation": translation, "inference_time": time.time() - start}

def translate_fa_to_en(text):
    if not text.strip():
        return {"error": "Input text cannot be empty"}
    if not is_dari(text):
        return {"error": "Input text must be in Dari/Persian only"}

    mbart_tokenizer.src_lang = "fa_IR"
    inputs = mbart_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    inputs["forced_bos_token_id"] = mbart_tokenizer.lang_code_to_id["en_XX"]
    start = time.time()
    output = mbart_model.generate(**inputs)
    translation = mbart_tokenizer.decode(output[0], skip_special_tokens=True)
    return {"translation": translation, "inference_time": time.time() - start}

# MT5
def mt5_translate_en_to_fa(text):
    if not text.strip():
        return {"error": "Input text cannot be empty"}
    if not is_english(text):
        return {"error": "Input text must be in English only"}

    inputs = mt5_en_fa_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    start = time.time()
    output = mt5_en_fa_model.generate(**inputs)
    translation = mt5_en_fa_tokenizer.decode(output[0], skip_special_tokens=True)
    return {"translation": translation, "inference_time": time.time() - start}

def mt5_translate_fa_to_en(text):
    if not text.strip():
        return {"error": "Input text cannot be empty"}
    if not is_dari(text):
        return {"error": "Input text must be in Dari/Persian only"}

    inputs = mt5_fa_en_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    start = time.time()
    output = mt5_fa_en_model.generate(**inputs)
    translation = mt5_fa_en_tokenizer.decode(output[0], skip_special_tokens=True)
    return {"translation": translation, "inference_time": time.time() - start}

def main():
    print("English ⇄ Dari Translator")
    language = input("Translate from (en/fa): ").strip().lower()
    if language not in ("en", "fa"):
        print("Invalid direction! Use 'en' or 'fa'.")
        return

    text = input(f"Enter text in {language}: ").strip()
    print("\n--- Translations ---")

    if language == "en":
        print("MBART:", translate_en_to_fa(text))
        print("MT5  :", mt5_translate_en_to_fa(text))
    else:
        print("MBART:", translate_fa_to_en(text))
        print("MT5  :", mt5_translate_fa_to_en(text))

if __name__ == "__main__":
    main()
