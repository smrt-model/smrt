#
# format the smrt code in the standard way. This script should be run before every commit to ensure correct formatting
#
pixi run --frozen ruff format .
pixi run --frozen ruff check --select I --fix .
pixi run --frozen ruff check --fix .
