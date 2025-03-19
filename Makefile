all: run

run: 
	uv run train.py

tok:
	uv run tokenizer.py

download:
	uv run download_data.py