#!/usr/bin/env python

import os
import sys
import tempfile
from neurotools.plotting import plot
import matplotlib.pyplot as plt

# Read in passed file
try:
    file = list(sys.argv)[1]
except IndexError:
    raise RuntimeError('A file must be specified!')

# Parse optional space argument
space = None
if len(list(sys.argv)) == 3:
    space = list(sys.argv)[2]

# Gen temp file
tf = tempfile.NamedTemporaryFile(suffix='.png')

# Plot and save
plot(file, space=space)
plt.savefig(tf.name, dpi=100,
            transparent=True,
            bbox_inches='tight')

# Open with viu
os.system('viu ' + tf.name)

# Close file, deleting
tf.close()
