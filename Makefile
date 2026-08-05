.PHONY: test smoke

test:
	pytest -q

smoke:
	python -m paper_v2.smoke --per-axis 10

