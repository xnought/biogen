all: run

run: 
	uv run main.py

tok:
	uv run tokenizer.py

download:
	uv run download_data.py