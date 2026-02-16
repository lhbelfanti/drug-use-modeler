import json
import os

NOTEBOOKS_DIR = 'notebooks'
TARGET_NB = '08_feed_forward.ipynb'

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
            source = "".join(cell['source'])
            if "acc_standard = train_ffn(" in source:
                # Replace the whole block with correct calls
                # train_ffn(variation_name, w2v_dir, output_dir)
                new_source = [
                    "acc_standard = train_ffn(\"Standard\", f\"{MODELS_DIR_BASE}/word2vec/standard\", f\"{MODELS_DIR_BASE}/ffn/standard\")\\n",
                    "\n",
                    "acc_irony = train_ffn(\"Irony\", f\"{MODELS_DIR_BASE}/word2vec/irony\", f\"{MODELS_DIR_BASE}/ffn/irony\")\\n",
                    "\n",
                    "acc_obfuscated = train_ffn(\"Obfuscated\", f\"{MODELS_DIR_BASE}/word2vec/obfuscated\", f\"{MODELS_DIR_BASE}/ffn/obfuscated\")\\n"
                ]
                cell['source'] = new_source
                updated = True
                break

    if updated:
        with open(filepath, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f"Fixed paths in {TARGET_NB}")
    else:
        print(f"No paths needed fixing in {TARGET_NB}")

if __name__ == "__main__":
    fix_notebook()
