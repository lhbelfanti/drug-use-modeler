import json
import os

NOTEBOOKS_DIR = 'notebooks'
TARGET_NBS = ['08_feed_forward.ipynb', '09_cnn.ipynb', '10_rnn.ipynb']

def fix_notebook(filename):
    filepath = os.path.join(NOTEBOOKS_DIR, filename)
    if not os.path.exists(filepath):
        print(f"{filename} not found")
        return

    with open(filepath, 'r') as f:
        nb = json.load(f)
    
    updated = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            cell_updated = False
            for line in cell['source']:
                if ".int().numpy()" in line:
                    # y_pred = (preds >= 0.5).int().numpy()
                    # -> y_pred = np.array((preds >= 0.5).int().tolist())
                    
                    # Regex replacement might be safer but precise replacement for this known pattern is okay
                    # Pattern: variable = (preds >= threshold).int().numpy()
                    # We can just replace .int().numpy() with .int().tolist()) and wrap with np.array(...)
                    # But wrapping matches parsing the whole line.
                    
                    # Simpler: replace .numpy() with .tolist() and then wrap the assignment if possible?
                    # Or just:
                    # y_pred = (preds >= 0.5).int().numpy()
                    # becomes
                    # y_pred = np.array((preds >= 0.5).int().tolist())
                    
                    if "y_pred =" in line and "numpy()" in line:
                        parts = line.split("=")
                        lhs = parts[0]
                        rhs = parts[1]
                        rhs = rhs.replace(".numpy()", ".tolist()")
                        new_line = f"{lhs}= np.array({rhs.strip()})\\n"
                        new_source.append(new_line)
                        cell_updated = True
                    else:
                        new_source.append(line)
                else:
                    new_source.append(line)
            
            if cell_updated:
                cell['source'] = new_source
                updated = True

    if updated:
        with open(filepath, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f"Fixed numpy conversion in {filename}")
    else:
        print(f"No numpy conversion needed fixing in {filename}")

if __name__ == "__main__":
    for nb in TARGET_NBS:
        fix_notebook(nb)
