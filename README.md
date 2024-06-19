# Stochastic Generalized Ekman-Stokes model
This code is utilized to reproduce the numerical results presented in the recently submitted paper titled "A Generalized Stochastic Formulation of the Ekman-Stokes Model with Statistical Analyses."

Copyright 2023 Long Li.

## Getting Started

### Dependencies

* Prerequisites: Pytorch, Numpy, Scipy, netCDF4, Matplotlib.

* Tested with Intel CPUs and Tesla V100-PCIE-32GB GPU.

### Installing

```
git clone https://github.com/matlong/SGESM.git
```

### Executing program

* To run the model:
```
python3 run.py
```

* To diagnose the model from output data after run:
```
python3 diag.py
```

* The jupyter notebook `plot_figures.ipynb` is used to reproduce all figures in the submitted paper from the diagnostic data. 

<!---
## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```
-->

## Authors

Contact: long.li@inria.fr

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

<!---
Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
-->
