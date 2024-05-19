set positional-arguments

alias u := update

default:
  @just --list

@test *args="":
  poetry run pytest --pyargs memejax

update:
  # TODO(2): unexclude numpy
  poetry run poetry up --latest --exclude=numpy
  poetry update
  pre-commit autoupdate
  pre-commit run --all-files
