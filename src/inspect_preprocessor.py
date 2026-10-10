"""Print details about Sneha's saved Experiment A preprocessor."""
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "models" / "processed_experiment_A.joblib"
DEFAULT_PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor_experiment_A.joblib"

data = joblib.load(DATA_PATH)
path_value = data.get("preprocessor_path")
preprocessor_path = Path(str(path_value)) if path_value else DEFAULT_PREPROCESSOR_PATH
if not preprocessor_path.exists():
    preprocessor_path = DEFAULT_PREPROCESSOR_PATH

preprocessor = joblib.load(preprocessor_path)

print("Preprocessor path:", preprocessor_path)
print("Preprocessor type:", type(preprocessor))
print("Original input columns:", len(data.get("original_feature_columns", [])))
print("Numeric columns:", len(data.get("numeric_columns", [])))
print("Categorical columns:", len(data.get("categorical_columns", [])))
print("Saved feature metadata count:", len(data.get("feature_names", [])))

if hasattr(preprocessor, "transformers_"):
    print("\nTransformers:")
    for name, transformer, columns in preprocessor.transformers_:
        if isinstance(columns, str):
            column_list = [columns]
        else:
            try:
                column_list = list(columns)
            except TypeError:
                column_list = [columns]
        print(f"\n{name}: {type(transformer).__name__}")
        print("Columns:", column_list)
        if hasattr(transformer, "categories_"):
            for column, categories in zip(column_list, transformer.categories_):
                print(f"  {column}: {list(categories)}")

if hasattr(preprocessor, "get_feature_names_out"):
    try:
        names = list(preprocessor.get_feature_names_out())
        print("\nPreprocessor output feature count:", len(names))
        for name in names:
            print(name)
    except Exception as exc:
        print("\nCould not get output feature names:", repr(exc))
