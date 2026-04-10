"""
Inference using Helsinki-NLP pretrained MarianMT models (Transformer architecture).
No training required — models are downloaded automatically from HuggingFace.

Supported directions:
    zh-en : Chinese  -> English   (Helsinki-NLP/opus-mt-zh-en)
    en-zh : English  -> Chinese   (Helsinki-NLP/opus-mt-en-zh)
    en-fr : English  -> French    (Helsinki-NLP/opus-mt-en-fr)
    fr-en : French   -> English   (Helsinki-NLP/opus-mt-fr-en)

Usage:
    # Interactive mode
    python inference.py --direction zh-en

    # Translate a single sentence
    python inference.py --direction zh-en --text "你好，世界！"

    # Translate a file (one sentence per line)
    python inference.py --direction zh-en --input sentences.txt --output translations.txt
"""
import argparse
from transformers import MarianMTModel, MarianTokenizer

MODELS = {
    "zh-en": "Helsinki-NLP/opus-mt-zh-en",
    "en-zh": "Helsinki-NLP/opus-mt-en-zh",
    "en-fr": "Helsinki-NLP/opus-mt-en-fr",
    "fr-en": "Helsinki-NLP/opus-mt-fr-en",
}


def load_model(direction: str):
    model_name = MODELS[direction]
    print(f"Loading {model_name} ...")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.eval()
    print("Ready.")
    return tokenizer, model


def translate(texts: list[str], tokenizer, model, batch_size: int = 16) -> list[str]:
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        out = model.generate(**tokens, num_beams=4, max_length=512)
        results.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", type=str, default="zh-en", choices=list(MODELS.keys()))
    parser.add_argument("--text", type=str, default=None, help="Single sentence to translate")
    parser.add_argument("--input", type=str, default=None, help="Input file (one sentence per line)")
    parser.add_argument("--output", type=str, default=None, help="Output file for translations")
    args = parser.parse_args()

    tokenizer, model = load_model(args.direction)

    if args.text:
        result = translate([args.text], tokenizer, model)
        print(result[0])

    elif args.input:
        with open(args.input, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        results = translate(lines, tokenizer, model)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write("\n".join(results))
            print(f"Saved {len(results)} translations to {args.output}")
        else:
            for src, tgt in zip(lines, results):
                print(f"  {src}")
                print(f"  -> {tgt}\n")

    else:
        # Interactive mode
        src_lang, tgt_lang = args.direction.split("-")
        print(f"Interactive translation ({src_lang} -> {tgt_lang}). Empty line to quit.\n")
        while True:
            text = input("Input: ").strip()
            if not text:
                break
            result = translate([text], tokenizer, model)
            print(f"Output: {result[0]}\n")
