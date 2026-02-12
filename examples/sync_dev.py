import json

API_KEY = "9e293a83-feb0-4275-b6f0-540ba935b4bb"

for name in ["api_workflow", "local_workflow", "inverse_design_workflow"]:
    with open(f"examples/{name}.ipynb") as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        # Handle both string and list source formats
        src = cell["source"]
        if isinstance(src, str):
            lines = src.split("\n")
        else:
            lines = src

        new_lines = []
        for line in lines:
            if "from google.colab import userdata" in line:
                continue
            line = line.replace("userdata.get('HYPERWAVE_API_KEY')", f'"{API_KEY}"')
            line = line.replace('userdata.get("HYPERWAVE_API_KEY")', f'"{API_KEY}"')
            new_lines.append(line)

        if isinstance(src, str):
            cell["source"] = "\n".join(new_lines)
        else:
            cell["source"] = new_lines

    with open(f"examples/{name}_dev.ipynb", "w") as f:
        json.dump(nb, f, indent=1)

    # Verify
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            s = cell["source"]
            text = s if isinstance(s, str) else "".join(s)
            if "userdata" in text:
                raise Exception(f"{name}_dev code cell still has userdata!")

    print(f"Synced {name}_dev.ipynb")
