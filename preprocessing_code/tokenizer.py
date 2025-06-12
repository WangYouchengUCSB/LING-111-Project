import os
import spacy
import json
from multiprocessing import Pool, cpu_count

def get_verb_label(token):
    if token.pos_ != "VERB":
        return None

    has_dobj = any(child.dep_ == "dobj" for child in token.children)
    has_indirect = any(child.dep_ in {"iobj", "obl", "prep", "dative"} for child in token.children)

    if has_dobj and has_indirect:
        return "ditransitive"
    elif has_dobj:
        return "transitive"
    return None

def preprocess_utterance(doc):
    labels = set()
    for token in doc:
        label = get_verb_label(token)
        if label:
            labels.add(label)
    return list(labels)

def process_file(args):
    file_path, max_len = args
    nlp = spacy.load("en_core_web_sm")

    base = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(os.path.dirname(file_path), base + "_output.json")

    data = {"transitive": [], "ditransitive": []}
    skipped = 0

    with open(file_path, 'r', encoding='utf-8') as infile:
        # Using nlp.pipe for faster batch processing
        lines = [line.strip() for line in infile if line.strip()]
        if max_len:
            lines = [line for line in lines if len(line.split()) <= max_len]

        skipped = 0
        if max_len:
            skipped = sum(1 for line in lines if len(line.split()) > max_len)

        for doc, utterance in zip(nlp.pipe(lines), lines):
            labels = preprocess_utterance(doc)
            for label in labels:
                tokens = [token.text.lower() for token in doc]
                data[label].append(tokens)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=2)

    print(f"{os.path.basename(file_path)}: {len(data['transitive'])} transitive, {len(data['ditransitive'])} ditransitive. Skipped: {skipped}")

def label_files_in_folder_parallel(folder_path: str, max_len: int = None):
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]

    # Prepare arguments list for the pool
    args = [(file_path, max_len) for file_path in files]

    # Use number of CPUs or less if you want
    num_workers = min(cpu_count(), len(files))

    with Pool(num_workers) as pool:
        pool.map(process_file, args)

if __name__ == "__main__":
    label_files_in_folder_parallel("preprocessed_data")
