# crispy-fortnight

## Create the environment
1. Make sure that Python 3.11 is installed. You can use `pyenv` to make it locally like this:
```bash
pyenv local 3.11.10
```

2. Initialise a new virtual environment
```bash
python3 -m venv .venv
```

or, if using pyenv:
```bash
pyenv exec python3 -m venv .venv
```

and then source it
```bash
source .venv/bin/activate
```

3. Install the packages from requirements
```bash
pip install -r requirements.txt
```

If installing any new packages, update the file `requirements.txt` like this
```bash
pip freeze > requirements.txt
```
