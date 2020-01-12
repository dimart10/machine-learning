# Machine Learning algorithms in Python
Welcome! This repository is all about machine learning algorithms, using Python standard libraries such as numpy, matplotlib or scipy. Take a look around if you are interested, you can also [contribute](#contributing) if you feel like it.

## Getting Started
Keep reading if you want to know how to run any of the scripts.

### Prerequisites

- Any Python 3.X installation (*virtualenv* can come in handy to avoid messing with your OS packages).
- Python 3 standard libraries. It depends on the script you want to execute, but generally you'd want to have at least `numpy`, `matplotlib`, `pandas` and `scipy`.
- More advanced scripts may also use `scikit-learn`, `opencv` & `pillow`.

### Installing

Once you have your prerequisites installed:

- Download or clone this repository.
- Open a terminal at the folder of the script you want to run.
- Enter `python nameOfTheScript.py` or `python3 nameOfTheScript.py`.

If you run into any trouble, check you have installed all the packages included in the script. If that's
not the problem, please [open an issue or submit a pull request with the fix](#contributing).

### Data format

Every algorithm has its own main function with an example data-set & its
training/evaluation. But if you want, you can use your own data. To do so, you
must know that the `train()` function will always receive at least two numpy arrays:
first (*X*) a 2D matrix with 1 row for each training example, with each column representing
each of its attributes; and second (*Y*) a vector (one dimensional array) with the
index of the classification group each example belongs to. If yours is not a classification
problem, then it's just the real output value of each training example.

For example, if you are classifying fruit images you can use 0 to represent apple,
1 for banana and 2 for pear. Also, each row would contain the [flattened] information
of each one of the data-set images.


## Built With

* [Python 3](https://www.python.org/downloads/)
* [Numpy](https://numpy.org/)
* [Scipy](https://www.scipy.org/)
* [Matplotlib](https://matplotlib.org/)
* [Pandas](https://pandas.pydata.org/)
* [Scikit Learn](https://scikit-learn.org/stable/)
* [OpenCV](https://opencv.org/)
* [Pillow](https://pillow.readthedocs.io/en/stable/)
* [Kaggle](https://www.kaggle.com/) (for datasets)

## Contributing

Thanks for wanting to contribute to the project! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* [**Sergio Abreu García**](https://sag-dev.com)
* [**Diego Martínez Simarro**](https://github.com/dimart10)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

This project started as a college work repository for the subject *Aprendizaje Automático y Minería de Datos* (Machine Learning & Data Mining):

* [Universidad Complutense de Madrid](https://informatica.ucm.es/)
* [Pedro Antonio González Calero](http://gaia.fdi.ucm.es/people/pedro/) (subject teacher)
