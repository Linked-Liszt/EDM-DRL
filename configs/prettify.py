import json
import sys

assert len(sys.argv) >= 2
j = json.load(open(sys.argv[1]))
confirm = input(f'Overwrite "{sys.argv[1]}"? [y/n]: ')
if confirm and confirm[0] in {'y', 'Y'}:
  with open(sys.argv[1], 'w') as f:
    f.write(json.dumps(j, sort_keys=True, indent=2))
