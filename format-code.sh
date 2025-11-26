#
# format the smrt code in the standard way. This script should be run before every commit to ensure correct formatting
#
ruff format .
ruff check --select I --fix .
ruff check --fix .
