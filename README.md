# DiGriPy
District Heating Grid Simulation in Python

## Installation
For installation, you can clone this repository and use pip for solving dependencies and installation.
Please use Python3 <3.9, as CoolProp fails to install on Python 3.9.
```
python -m pip install digripy
```

You can alternatively manually clone this repo and then
```
python setup.py install
```
in the cloned directory.

## Running
For Running DiGriPy you can simply download the `tests` folder provided in the [git repository](https://github.com/lvorspel/DiGriPy) and run 
```
python -m digripy.run -i tests/example80C -l 20
```

in a command line. This will run the first 20 time steps of the medium temperature scenario and display the results in a browser. 

## Output
Results will not only be displayed in the browser, but also will be saved in your home directory in a folder called `digripy_results`. There you can also find spreadsheets with several results.
