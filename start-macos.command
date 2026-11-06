#!/bin/bash
cd "$(dirname "$0")" || exit 1
python3 start.py "$@"
result=$?
if [ "$result" -ne 0 ]; then
  echo "Startup failed. Please read the message above."
  read -r -p "Press Enter to close..."
fi
exit "$result"
