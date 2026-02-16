import json
import os

NOTEBOOKS_DIR = 'notebooks'

def fix_notebook(filename):
    filepath = os.path.join(NOTEBOOKS_DIR, filename)
    if not os.path.exists(filepath):
        return

    with open(filepath, 'r') as f:
        nb = json.load(f)
    
    updated = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            cell_updated = False
            for line in cell['source']:
                # init line length
                original_len = len(line)
                
                # Check for double escaped newline at the end
                if line.endswith('\\n'):
                    # Replace the last occurrence of \\n with \n
                    # Check if it's actually \\n and not \\\n (escaped backslash then newline)
                    # But simplest is:
                    new_line = line[:-2] + '\n'
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
        print(f"Fixed syntax in {filename}")
    else:
        print(f"No syntax issues found in {filename}")

if __name__ == "__main__":
    if not os.path.exists(NOTEBOOKS_DIR):
        print(f"Directory {NOTEBOOKS_DIR} not found.")
    else:
        notebooks = [f for f in os.listdir(NOTEBOOKS_DIR) if f.endswith('.ipynb')]
        for nb in notebooks:
            fix_notebook(nb)
