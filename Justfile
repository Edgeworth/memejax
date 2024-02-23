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
  SETUPTOOLS_USE_DISTUTILS=stdlib pre-commit run --all-files
