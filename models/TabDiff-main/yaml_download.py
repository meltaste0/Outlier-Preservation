import os
import yaml

with open("tabdiff.yaml") as f:
    env_data = yaml.safe_load(f)

for dep in env_data.get("dependencies", []):
    if isinstance(dep, str):
        os.system(f"pip install {dep}")
    elif isinstance(dep, dict) and "pip" in dep:
        for pkg in dep["pip"]:
            os.system(f"pip install {pkg}") 