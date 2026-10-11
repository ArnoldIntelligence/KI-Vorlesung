.PHONY: build-nb build-py clean-nb normalize-nb

build-nb:
	@for f in src/*.py; do \
		b="$$(basename "$$f")"; \
		jupytext --to notebook "$$f" -o "$${b%.py}.ipynb"; \
	done

build-py: normalize-nb
	@for f in *.ipynb; do \
		jupyter trust "$$f" >/dev/null; \
		jupytext --to py:light "$$f" -o "src/$${f%.ipynb}.py"; \
	done

normalize-nb:
	@JT_PY="$$(dirname "$$(command -v jupytext)")/python"; \
	"$$JT_PY" ../scripts/normalize_notebooks.py .

clean-nb:
	@rm *.ipynb
