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
                # Check for the broken line pattern
                if "y_pred = np.array((preds >)" in line or "y_pred = np.array((preds >" in line:
                    # Fix it to the correct logic
                    # Original was y_pred = (preds >= 0.5).int().numpy()
                    # We want y_pred = np.array((preds >= 0.5).int().tolist())
                    new_line = "    y_pred = np.array((preds >= 0.5).int().tolist())\\n"
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
        print(f"Fixed broken syntax in {filename}")
    else:
        print(f"No broken syntax found in {filename}")

if __name__ == "__main__":
    for nb in TARGET_NBS:
        fix_notebook(nb)
