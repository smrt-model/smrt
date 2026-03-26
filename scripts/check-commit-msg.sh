#!/bin/sh
#
# An example hook script to check the commit log message.
# Called by "git commit" with one argument, the name of the file
# that has the commit message.  The hook should exit with non-zero
# status after issuing an appropriate message if it wants to stop the
# commit.  The hook is allowed to edit the commit message file.
#
# To enable this hook, rename this file to "commit-msg".

# Uncomment the below to add a Signed-off-by line to the message.
# Doing this in a hook is a bad idea in general, but the prepare-commit-msg
# hook is more suited to it.
#
# SOB=$(git var GIT_AUTHOR_IDENT | sed -n 's/^\(.*>\).*$/Signed-off-by: \1/p')
# grep -qs "^$SOB" "$1" || echo "$SOB" >> "$1"

# This example catches duplicate Signed-off-by lines.

test "" = "$(grep '^Signed-off-by: ' "$1" |
	 sort | uniq -c | sed -e '/^[ 	]*1[ 	]/d')" || {
	echo >&2 Duplicate Signed-off-by lines.
	exit 1
}


# Read the commit message from the file provided by Git
commit_msg_file="$1"
commit_msg=$(cat "$commit_msg_file")

# Define a regex pattern to check for imperative mood
# This pattern checks if the message starts with a verb in imperative form
pattern="^(add|fix|update|remove|refactor|improve|change|create|delete|document|optimize|revert|bump|clean|upgrade|downgrade|move|rename|replace|secure|simplify|test|upgrade|use|verify)[ :]"

# Check if the commit message matches the pattern
if ! echo "$commit_msg" | grep -iqE "$pattern"; then
    echo "❌ Commit message must be in imperative mood and start with a verb like 'Add', 'Fix', 'Update', etc."
    echo "Example: 'Add feature X' instead of 'Adding feature X' or 'Added feature X'."
    exit 1
fi

exit 0
