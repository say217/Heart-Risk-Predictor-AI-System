import importlib

packages = {
    "flask": "Web framework for building the app",
    "gunicorn": "Production WSGI server (for running in Docker/prod)",
    "joblib": "Loading the saved scikit-learn pipeline/model",
    "pandas": "Data manipulation (creating DataFrame from input)",
    "numpy": "Numerical operations (used internally by sklearn & pandas)",
    "sklearn": "Your trained multiclass classification pipeline"
}

print("Package,Version,Purpose")
for pkg, purpose in packages.items():
    module = importlib.import_module(pkg)
    print(f"{pkg},{module.__version__},{purpose}")












