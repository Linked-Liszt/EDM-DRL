import json
import sys
from earl.runner import EvoACRunner

def main():
  assert len(sys.argv) >= 2
  config = json.load(open(sys.argv[1]))
  runner = EvoACRunner(config)
  runner.train()

if __name__ == '__main__':
  main()
