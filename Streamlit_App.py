import streamlit as st
import PyPDF2
import os
import tempfile
import sentencepiece as spm
import ctranslate2
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
from tempfile import NamedTemporaryFile
import re

#Translation Function
def translate(source, translator, sp_source_model, sp_target_model):

    source_sentences = sent_tokenize(source)
    source_tokenized = sp_source_model.encode(source_sentences, out_type=str)
    translations = translator.translate_batch(source_tokenized)
    translations = [translation[0]["tokens"] for translation in translations]
    translations_detokenized = sp_target_model.decode(translations)
    translation = " ".join(translations_detokenized)

    return translation


# [Modify] File paths here to the CTranslate2 SentencePiece models.
ct_model_path = "ct2_model/"
sp_source_model_path = "source.model"
sp_target_model_path = "target.model"

# Create objects of CTranslate2 Translator and SentencePieceProcessor to load the models
translator = ctranslate2.Translator(ct_model_path, "cpu")    # or "cuda" for GPU
sp_source_model = spm.SentencePieceProcessor(sp_source_model_path)
sp_target_model = spm.SentencePieceProcessor(sp_target_model_path)


# Define patterns commonly found in reference sections
reference_patterns = [
    "references",
    "bibliography",
    "citations",
    "acknowledgments",
    "appendix",
    "appendices",
]

# Function to check if a sentence appears to be a reference
def is_reference(sentence):
    for pattern in reference_patterns:
        if pattern in sentence.lower():
            return True
    return False


# Text Extraction Function
def extractText(file_path):
    pdfFileObj = open(file_path, "rb")
    pdfReader = PyPDF2.PdfReader(pdfFileObj)
    num_pages = len(pdfReader.pages)

    text = ""
    for page_num in range(num_pages):
        pageObj = pdfReader.pages[page_num]
        text += pageObj.extract_text()

    pdfFileObj.close()
    return text


# Summarizer Function
def summarize(text):
    # Process text by removing numbers and unrecognized punctuation
    processedText = re.sub("’", "'", text)
    processedText = re.sub("[^a-zA-Z' ]+", " ", processedText)
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(processedText)

    # Normalize words with Porter stemming and build word frequency table
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue
        elif stemmer.stem(word) in freqTable:
            freqTable[stemmer.stem(word)] += 1
        else:
            freqTable[stemmer.stem(word)] = 1

    # Normalize every sentence in the text
    sentences = sent_tokenize(text)
    stemmedSentences = []
    sentenceValue = dict()
    for sentence in sentences:
        stemmedSentence = []
        for word in sentence.lower().split():
            stemmedSentence.append(stemmer.stem(word))
        stemmedSentences.append(stemmedSentence)

    # Calculate value of every normalized sentence based on word frequency table
    for num in range(len(stemmedSentences)):
        for wordValue in freqTable:
            if wordValue in stemmedSentences[num]:
                if sentences[num][:12] in sentenceValue:
                    sentenceValue[sentences[num][:12]] += freqTable.get(wordValue)
                else:
                    sentenceValue[sentences[num][:12]] = freqTable.get(wordValue)

    # Determine average value of a sentence in the text
    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue.get(sentence)

    average = int(sumValues / len(sentenceValue))

    # Create summary of text excluding reference-like sentences
    summary = ""
    for sentence in sentences:
        if sentence[:12] in sentenceValue and sentenceValue[sentence[:12]] > (3.0 * average) and not is_reference(sentence):
            summary += " " + " ".join(sentence.split())
            
    return summary


# Title for page
st.set_page_config(page_title="Article Summarizer", page_icon="🤖")
# Header
st.title("Article Summarizer")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Read pdf file
    with st.spinner('Loading...'):
        with NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            file_path = tmp_file.name

    
    with st.spinner('Loading...'):            
                
        pdf_text = extractText(file_path)
        summary_text = summarize(pdf_text)
        # Display Summary
        source_text = st.text_area("Summary", value=summary_text, height=400)
        translation = translate(source_text, translator, sp_source_model, sp_target_model)
        # Button for downloading Translated Summary
        download = st.download_button("Download Translated Summary", translation, "Translated Summary.txt")
                        
else:
    st.warning("Please upload a PDF file.")
