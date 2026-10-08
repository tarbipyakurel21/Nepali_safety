.PHONY: check test artifact shell

check: test artifact shell

test:
	python3 -m unittest discover -s tests -v

artifact:
	python3 scripts/check_artifact.py --include-working-tree

shell:
	@find scripts -type f \( -name '*.sh' -o -name '*.sbatch.sh' \) -print0 | xargs -0 -n1 bash -n
