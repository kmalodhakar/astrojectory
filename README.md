# ECE 143 Group 6: Astrojectory

## Collaborators

- Ruining Feng
- Andrea Kang
- Haozhe Liu
- Ameya Mandale
- Kunaal Malodhakar

<img src="https://github.com/kmalodhakar/astrojectory/blob/master/gifs/hazardous_astro.gif" width="400" height="400" />

## Objective
To investigate the nuances and patterns across a multitude of asteroids in the solar system, ranging from hazard classification to trajectory prediction.

***
## Datasets

- [Open Asteroid Dataset](https://www.kaggle.com/basu369victor/prediction-of-asteroid-diameter?select=Asteroid_Updated.csv)
- [NASA Asteroid Classification Dataset](https://www.kaggle.com/shrutimehta/nasa-asteroids-classification)

***

### Dependencies
Python 3.6+ is required.
- Pandas
- Sklearn
- Numpy
- Matplotlib
- Seaborn

## Installation Instructions

1) Clone the repository:
```
git clone https://github.com/kmalodhakar/astrojectory.git
cd astrojectory
```

2) Download Asteroid_Updated.csv from the [Open Asteroid Dataset](https://www.kaggle.com/basu369victor/prediction-of-asteroid-diameter?select=Asteroid_Updated.csv)
3) Run cleaner.py to obtain Asteroid_au_to_km.csv:
```
python3 scripts/cleaner.py
```
***

## Visualization

For dataset analysis visualization, the two scripts can be used:

- classify_heatmaps
- eda_from_csv

Both of the scripts are in .ipynb and .py files. 
While .ipynb may be preferred, the commands for .py are below:

```
python3 classify_heatmaps.py
python3 eda_from_csv.py
```

For regression, run the following command:
```
python3 regression.py 
```

For trajectory, run the following notebook:
```
Trajectory Visualization.ipynb
```

## License

MIT License - see the LICENSE file for further information.