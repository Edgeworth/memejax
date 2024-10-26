set positional-arguments

alias u := update

default:
  @just --list

@test *args="":
  poetry run pytest --pyargs memejax

update:
  poetry run poetry up --latest
  poetry update
  pre-commit autoupdate

check:
  poetry check
  poetry run mypy --install-types --non-interactive
  pre-commit run --all-files

fix:
  pre-commit run --all-files
