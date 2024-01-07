import sys
import os
import re
from glob import glob

path = sys.argv[1]
split = sys.argv[2]
pattern = sys.argv[3]

pat = re.compile(pattern)
all_files = {}
for file in glob(os.path.join(path, "*best*")):
    root, rest = file.split(split)
    if root not in all_files:
        all_files[root] = []
    value = float(pat.search(file).group())
    all_files[root].append([file, value])

for k in all_files:
    worst = sorted(all_files[k], key=lambda x: -x[1])[1:]
    worst = [x[0] for x in worst]
    print("\n".join(worst))
