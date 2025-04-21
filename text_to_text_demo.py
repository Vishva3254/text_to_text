import warnings
# warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="You are using the default legacy behaviour.*")
warnings.filterwarnings("ignore", message="The sentencepiece tokenizer.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`.*")
warnings.filterwarnings("ignore", message="Tried to instantiate class '__path__._path'.*")

# TO GET THE ARGOTRANSLTE PACKAGE
import argostranslate.package

argostranslate.package.update_package_index()

available_packages = argostranslate.package.get_available_packages()

package_to_install = next(
    (pkg for pkg in available_packages if pkg.from_code == "en" and pkg.to_code == "fa"), None
)

if package_to_install:
    download_path = package_to_install.download()
    argostranslate.package.install_from_path(download_path)
    print(" Argos language package installed!")
else:
    print(" English to Farsi package not found.")

import streamlit as st
import json
import os
from translate_using_model import translate_en_to_fa, translate_fa_to_en, mt5_translate_en_to_fa, mt5_translate_fa_to_en
from translate_using_library import translate_googletrans, translate_deeptranslator, translate_argos

EN_TO_FA_FILE = "en_to_fa.json"
FA_TO_EN_FILE = "fa_to_en.json"

def save_to_json(input_text, translations, direction):
    file_path = EN_TO_FA_FILE if direction == "en_to_fa" else FA_TO_EN_FILE
    entry = {"input": input_text, "translations": translations}
    
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

st.set_page_config(page_title="English ⇄ Dari Translator", layout="centered")
st.title("English ⇄ Dari Translation Demo")

direction = st.radio("Choose translation direction:", ["English to Dari", "Dari to English"])
is_en_to_fa = direction == "English to Dari"
src_lang = "en" if is_en_to_fa else "fa"
dst_lang = "fa" if is_en_to_fa else "en"

text = st.text_area(f"Enter text in {src_lang.upper()}:")

if st.button("Translate") and text.strip():
    st.subheader("Translation Results")
    results = {}

    if is_en_to_fa:
        results["MBART"] = translate_en_to_fa(text)
        results["MT5"] = mt5_translate_en_to_fa(text)
        results["Argos Translate"] = translate_argos(text, src=src_lang, dest=dst_lang)
    else:
        results["MBART"] = translate_fa_to_en(text)
        results["MT5"] = mt5_translate_fa_to_en(text)

    results["Googletrans"] = translate_googletrans(text, src=src_lang, dest=dst_lang)

    results["Deep Translator"] = translate_deeptranslator(text, src=src_lang, dest=dst_lang)

    for method, output in results.items():
        st.markdown(f"**{method}**")
        st.success(output["translation"])
        if output["inference_time"]:
            st.caption(f" Inference time: {output['inference_time']:.3f} sec")

    save_to_json(text, results, direction="en_to_fa" if is_en_to_fa else "fa_to_en")
    st.toast("Translation results saved!", icon="📁")
