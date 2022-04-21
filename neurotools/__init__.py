import neurotools_data
from pathlib import Path

# Use data from sep neurotools_data repo which
# is installed as req
data_dr = Path(neurotools_data.__file__).parent.absolute()

# Set version
__version__ = '0.22'