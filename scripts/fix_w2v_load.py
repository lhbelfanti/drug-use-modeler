import json
import os

NOTEBOOKS_DIR = 'notebooks'
TARGET_NB = '07_word2vec_embeddings.ipynb'

def fix_notebook():
    filepath = os.path.join(NOTEBOOKS_DIR, TARGET_NB)
    if not os.path.exists(filepath):
        print(f"{TARGET_NB} not found")
        return

    with open(filepath, 'r') as f:
        nb = json.load(f)
    
    updated = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            cell_updated = False
            for line in cell['source']:
                if "Word2Vec.load('../models/word2vec/standard/word2vec.model')" in line:
                    new_line = line.replace("'../models/word2vec/standard/word2vec.model'", "f'{MODELS_DIR_BASE}/word2vec/standard/word2vec.model'")
                    new_source.append(new_line)
                    cell_updated = True
                else:
                    new_source.append(line)
            
            if cell_updated:
                cell['source'] = new_source
                updated = True

    if updated:
        with open(filepath, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f"Fixed paths in {TARGET_NB}")
    else:
        print(f"No paths needed fixing in {TARGET_NB}")

if __name__ == "__main__":
    fix_notebook()
