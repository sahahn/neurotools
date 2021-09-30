# neurotools

Creative and original name? Maybe not. Lots of cool and helpful tools for working with neuroimaging data? Yes.

## What's Included?

neurotools is split into a number of different directories / sub-packages, each including different functionality (though key pieces from different sub-directories are shared allowing for some cool functionality!). Be low are some brief descriptions of the different sub-packages.

- loading : 
    This directory encompasses a set of utilities for loading generic neuroimaging data, as well as some ABCD study specific loading utilities.

- misc : 
    A set of misc. functions that don't fit in nicely anywhere else.

- plotting :
    Tools and functions for plotting surface representations of the data. Also
    smart functions for easy quick plots and collages of surface representations.
    Still a big work in progress as plotting can be messy, but in general these
    tools are designed to help make it easier and more stream-lined.

- random : 
    Different from misc... this includes functions that are based on randomness.
    These include utilities for generating constrained permutations based on block structures
    and also for generating random / null surface parcellations.

- rely :
    Included here are utilities for running 'reliability' like tests. These
    were featured in https://github.com/sahahn/ABCD_Consortium_Analysis

- stats :
    Featured here are wrappers and other code for running statical tools,
    from basic functions (i.e., calculating cohen's d effect size with or without NaN's)
    to running mixed linear models.

- transform :
    Tools for transforming data - that is to say, tools for extracting ROIs from surface
    level data are provided, with utility for inverse_transforming them as well. Also included,
    but still more as a work in progress, are some utilities for converting between different
    common surfaces, e.g., from fs LR 32k surface parcellations to a version without medial
    walls and with the sub-cortical regions included.


Besides these main functionalities, another main folder is included, called "slurm". This folder
will store examples for using the neurotools package on a SLURM cluster is a massively paralell way. Included are, and in the future will be, examples with extra scripts and pieces responsible for submitting and collecting jobs.

## Install

For now, this repository only lives on github, so the only way to download it is by cloning (`git clone https://github.com/sahahn/neurotools`) and pip installing (navigate into the directory then run `pip install .`) the repository.