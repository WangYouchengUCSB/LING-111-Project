import os
import spacy
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
    folder = os.path.dirname(file_path)
    output_path = os.path.join(folder, base + "_verbs.txt")

    selected_lines = []
    skipped = 0

    with open(file_path, 'r', encoding='utf-8') as infile:
        lines = [line.strip() for line in infile if line.strip()]
        if max_len:
            filtered_lines = []
            for line in lines:
                if len(line.split()) <= max_len:
                    filtered_lines.append(line)
                else:
                    skipped += 1
            lines = filtered_lines

        for doc, utterance in zip(nlp.pipe(lines), lines):
            labels = preprocess_utterance(doc)
            if labels:
                tokens = [token.text.lower() for token in doc]
                sentence = " ".join(tokens)
                selected_lines.append(sentence)

    with open(output_path, 'w', encoding='utf-8') as out_file:
        out_file.write("\n".join(selected_lines) + "\n" if selected_lines else "")

    print(f"{os.path.basename(file_path)}: {len(selected_lines)} labeled (trans/ditrans), Skipped: {skipped}")

def label_files_in_folder_parallel(folder_path: str, max_len: int = None):
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]

    args = [(file_path, max_len) for file_path in files]

    num_workers = min(cpu_count(), len(files))

    with Pool(num_workers) as pool:
        pool.map(process_file, args)

if __name__ == "__main__":
    label_files_in_folder_parallel("preprocessed_data")
