import sys
from funcs import run_from_save

# Unpack args
args = list(sys.argv[1:])
temp_dr = args[0]

# Get if short or not, if not passed assume not short
try:
    short = bool(int(args[1]))
except IndexError:
    short = False

# Submit to run
run_from_save(temp_dr, short=short)