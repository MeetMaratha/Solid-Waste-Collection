## Solid waste collection Optimization

### Requirements

- asttokens==2.0.5
- backcall==0.2.0
- cycler==0.11.0
- debugpy==1.6.0
- decorator==5.1.1
- entrypoints==0.4
- executing==0.8.3
- fonttools==4.33.3
- gurobipy==9.5.1
- ipykernel==6.13.0
- ipython==8.3.0
- jedi==0.18.1
- jupyter-client==7.3.1
- jupyter-core==4.10.0
- kiwisolver==1.4.2
- matplotlib==3.5.2
- matplotlib-inline==0.1.3
- nest-asyncio==1.5.5
- numpy==1.22.3
- opencv-python==4.5.5.64
- packaging==21.3
- pandas==1.4.2
- parso==0.8.3
- pexpect==4.8.0
- pickleshare==0.7.5
- Pillow==9.1.0
- prompt-toolkit==3.0.29
- psutil==5.9.0
- ptyprocess==0.7.0
- pure-eval==0.2.2
- Pygments==2.12.0
- pyparsing==3.0.9
- python-dateutil==2.8.2
- pytz==2022.1
- pyzmq==22.3.0
- six==1.16.0
- stack-data==0.2.0
- tornado==6.1
- traitlets==5.2.0
- wcwidth==0.2.5 

### General Idea
We are using linear programming optimization in gurobipy to solve for optimal garbage collection.

### Folder structure and files
- Chandigarh QGIS : It contains all files related to QGIS we have used.
- Data : It contains all data which we are using for our computation.
- Bin Locations.csv : They are randomly generated points in QGIS and clustered using K-Means. The depot is assigned ward -1.
- distances.csv : This contains the distance matrix of all points in same ward or from depot.
- Static Data : Contains data regarding Static cases
- Dynamic Data : Contains Data regarding Dynamic cases
- Visited Truck # : Visited truck list with fill ratio
- Truck # Data : Each truck data used for computation.
- Statistics.csv : Stats for the respective case.

### Python Files
- get_distance.py : Run this to get distance matrix
- static_function.py : Static Optimization code
- dynamic_function.py : Dynamic Optimization code
- static_unweighted.py : Static unweighted optimization code
- static_find_weights.py : Finding the wieghts for static case
- dynamic_unweighted.py : Dynamic unweighted optimization code
- dynamic_weight_finding.py : Finding weights for dynamic case. 
