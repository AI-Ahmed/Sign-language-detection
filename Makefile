setup:
  python3 -m -venv ~/.Sign-langunage-detection

install:
  pip install -r requirements.txt
  pip install nbval
  pip install .
  
test:
  python -m pytest --nbval Sign-Language-CNNs-Fine-Tuned.ipynb
lint:
  pylint --disable=R,C Sign-language-detection cli web
  
