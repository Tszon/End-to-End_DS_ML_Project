# scripts/combine_model_params.py
import os
import json
from src.myproject.utils.paths import MODELS_DIR

def combine_model_params(best_model_params_path, output_path):
    combined_params = {}
    for path in best_model_params_path:
        with open(path, "r") as f:
            params = json.load(f)
            model_name = path.stem.split("_best_params_")[0]
            combined_params[model_name] = params
    with open(output_path, "w") as f:
        json.dump(combined_params, f, indent=4)
    
    print(f"\nCombined model parameters saved to:\n- {output_path}")


if __name__ == "__main__":
    os.makedirs(MODELS_DIR / "params", exist_ok=True)

    best_model_params_path = [
    MODELS_DIR / "params" / "log_reg_best_params_v1.json",
    MODELS_DIR / "params" / "rf_best_params_v1.json",
    MODELS_DIR / "params" / "xgb_best_params_v1.json"
]
    
    combine_model_params(
        best_model_params_path,
        MODELS_DIR / "params" / "combined_model_params_v1.json"
        )
    