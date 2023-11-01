import ctranslate2
import sentencepiece as spm

# Set file paths
source_file_path = "run/source.vocab"
target_file_path = "run/target.vocab"

sp_source_model_path = "source.model"
sp_target_model_path = "target.model"

ct_model_path = "models/model.dutch_step_70000.pt/"


# Load the source SentecePiece model
sp = spm.SentencePieceProcessor()
sp.load(sp_source_model_path)

# Open the source file
with open(source_file_path, "r") as source:
  lines = source.readlines()

source_sents = [line.strip() for line in lines]

# Subword the source sentences
source_sents_subworded = sp.encode_as_pieces(source_sents)

# Translate the source sentences
translator = ctranslate2.Translator(ct_model_path, device="cuda")  # or "cuda" for GPU
translations = translator.translate_batch(source_sents_subworded, batch_type="tokens", max_batch_size=4096)
translations = [translation.hypotheses[0] for translation in translations]

# Load the target SentecePiece model
sp.load(sp_target_model_path)

# Desubword the target sentences
translations_desubword = sp.decode(translations)


# Save the translations to the a file
with open(target_file_path, "w+", encoding="utf-8") as target:
  for line in translations_desubword:
    target.write(line.strip() + "\n")

print("Done")
