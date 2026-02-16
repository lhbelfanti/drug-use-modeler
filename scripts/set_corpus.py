import json
import os
import re

NOTEBOOKS_DIR = 'notebooks'
NEW_CORPUS = 'raw_corpus'

def update_notebook(filename):
    filepath = os.path.join(NOTEBOOKS_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Skipping {filename}")
        return

    with open(filepath, 'r') as f:
        nb = json.load(f)
    
    updated = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            cell_updated = False
            for line in cell['source']:
                # Look for the specific configuration line
                if "CORPUS_NAME =" in line:
                    # Replace whatever is currently set with the new corpus
                    # Regex to match: CORPUS_NAME = 'anything' ...
                    new_line = re.sub(r"CORPUS_NAME = '.*?'", f"CORPUS_NAME = '{NEW_CORPUS}'", line)
                    new_source.append(new_line)
                    if new_line != line:
                        cell_updated = True
                else:
                    new_source.append(line)
            
            if cell_updated:
                cell['source'] = new_source
                updated = True
                # Break after finding and updating the config cell to avoid accidental edits elsewhere
                # Assuming config is in one cell.
                break 

    if updated:
        with open(filepath, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f"Updated {filename} to use {NEW_CORPUS}")
    else:
        print(f"No changes needed for {filename}")

if __name__ == "__main__":
    notebooks = [f for f in os.listdir(NOTEBOOKS_DIR) if f.endswith('.ipynb')]
    for nb in notebooks:
        update_notebook(nb)
