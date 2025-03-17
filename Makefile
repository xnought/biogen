all: run

run: 
	uv run main.py

download:
	uv run download_data.py