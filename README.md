# Solid waste collection Optimization

This repository presents a python code implementation of [Dynamic routing for efficient waste collection in resource constrained societies](https://www.nature.com/articles/s41598-023-29593-x.epdf?sharing_token=T7qW4Wf6IrrbA3hosV7ypNRgN0jAjWel9jnR3ZoTv0OQTYJ1chF4A3By2Het_SUUYeSL-_Okqgm1JVnX0j-EjhgwaZi5El6D-Pvva2z9OcSiK4n9vcfq7jj7wImjeKUjdDrwut6rQbS8rY-21_IeFMivA5PNCc3VJdJsjT17szo%3D)

## Abstract

Waste collection in developing nations faces multi-fold challenges, such as resource constraints and real-time changes in waste values, while finding the optimal routes. This paper attempts to address these challenges by modeling real-time waste values in smart bins and Collection Vehicles (CV). Further, waste value prioritized routes for coordinated CV, during various time intervals are modeled in a multi-agent environment for finding good routes. The CV, as agents, implement the formulated linear program to maximize the collected waste while minimizing the distance to the central depot. The city of Chandigarh, India, was divided into regions and the model was implemented to achieve significantly better performance in terms of waste collected in less distance and total bins covered when compared to the existing scenario. The stakeholders can use the outcomes to effectively plan the resources for better collection practices, which will have a positive impact on the environment.

## Inputs

1. **Agent:** Waste collection vehicle.
2. **Agent attributes:**
   - Waste fille percentage
   - Distance travelled
3. **Environment:**
   - Road Network
   - Number of bins and their positions
   - Depot position

## Outputs

Best path for each agent such that there is maximum amount of waste collection while travelling the least distance.

## Installation

1. Clone the github repository using the following command

```
git clone https://github.com/MeetMaratha/Solid-Waste-Collection.git
```

2. Make a virtual environment to work. This is only needed to separate the project from the main system install. You can use the following command in the cloned github folder to generate the virtual environment on Windows and Linux distributions other than Ubuntu/Debian

```
python -m venv solid_venv
```

For Ubuntu/Debian use following command

```
python3 -m venv solid_venv
```
3. Activate the virtual environment using the following command in the cloned github folder

**For Bash/Windows/ZSH shell:**
```
source solid_venv/bin/activate
```
**For Fish shell:**
```
source solid_venv/bin/activate.fish
```
4. Install the python packages required using the following command
```
pip install -r requirements.txt
```

## General flow 

1. We assume that a road network with placement of _**n**_ bins and 2 depot are already provided.
2. We then choose one of those depot as starting point for all the trucks.
3. In a simulation there can be _**m**_ number of trucks.
4. They all start at **t = 0**, where _**t**_ defines time.
5. They generate a route to collect maximum amount of garbage while travelling the least amount using an optimization function.
6. As time passes the amount of garbage in the bins changes, which changes the path that the **Collection Vehicle (CV)** should take.
7. If a CV collects garbage from a bin, its fill ratio is set to 0 percentage for the complete simulation from that time interval.
8. As a CV is filled to the maximum, it then returns to the depot. This return trip cost is also considered in the optimization function.

The following image displays an example of the whole work flow:

![](Images/Figure_001.png) 

## Optimization function

We are trying to optimize the following equation using the gurobipy module in python.

$$
Obj(Maximize)=\sum_{i,j \in A} w_1 X_{ij} C_{ij} - w_2 Y_i f_i * BT
$$

Following is the variables used in the optimization and their meaning

| Variables                     | Description                                                                            |
| :---:                         | :---:                                                                                  |
| $i, j \in D$                  | $i^{th}$ bin and $j^{th}$ bin respectively.                                            |
| $D$                           | Total bins for a particular region including depot.                                    |
| $N$                           | Bins for a particular CV excluding depot.                                              |
| $A$                           | Set of all the arcs formed from $i^{th}$ bin to $j^{th}$ bin for $\forall i ,j \in$ D. |
| $C_{ij}\in \mathbb{R}^+$      | Distance cost from bin $i$ to bin $j$ for a CV.                                        |
| $X_{ij} \in \{0, 1\}$         | Binary variable that is 1 if a CV is	travelling between the bin $i$ and bin $j$.      |
| $Y_{i} \in \{0, 1\}$          | A binary variable that is 1 if a CV has visited bin $i$.                               |
| $P_{t} \in [0.0, 100.0]$      | Cumulative fill percentage of CV for the time interval $t$.                            |
| $u_{i}  \in [0, 100 -P_{t}] $ | The fill percentage of the CV visiting bin $i$ for a specific time interval.           |
| $st \in D$                    | The starting bin of a CV in a new time interval.                                       |
| $w_{1} \in [0.0, 1.0]$        | The weight assigned to the distance in the objective function.                         |
| $w_{2} \in [0.0, 1.0]$        | The weight assigned to the waste amount in the objective function.                     |
| $f_{i} \in [0.0, 1.0]$        | The fill ratio of bin at bin $i$.                                                      |
| $BT \in \mathbb{R}^+$         | The conversion factor for fill of smart bin to fill of CV.                             |

There are constraints and such which we use to solve the problem defined, you can check them out by reading the paper.

## Experiment

The road network was extracted from [OpenStreetMap](https://www.openstreetmap.org) (OSM) database. OpenStreetMap API was then used to calculate the distance between all bins and generate a distance matrix used in the optimization process. We applied K-Means clustering algorithm to cluster these points in a fixed set of clusters provided in [QGIS](https://www.qgis.org/en/site/). In this experiment, we have selected total clusters to be three. The number can be updated based on user requirements. After clustering, region 1, region 2, and region 3 had 95, 111, and 94 points, respectively. The clusters were considered as three regions for which CV had to be allocated. To replicate the actual field activities, we have assumed that a CV starts and ends its route at the depot. A representation of the generated map is as follows:

![](Images/Figure_002.png)

## Case Studies
Experiments were carried out on an **AMD A6-9220 processor**, which runs at **2.5 GHz** and utilizes **8 GB RAM**. The model was implemented in **Python 3.10.5** and solved with Gurobi optimizer version **Gurobi 9.5.1**. A total of four scenarios were implemented. The minimum and maximum solving times were around 67.0618 s having total variable count of 7041 and total constraint count of 7075 for the maximum case. The suggestion of the decision makers (Municipal Corporation Chandigarh) was to provide equal preference to find minimum distance routes and maximize the waste collection. Hence, we assigned $w_1$ and $w_2$ to be **0.5** to execute the scenarios. This means that our optimization model gives equal importance to minimizing the distance and maximizing the waste collected. The maximum capacity of a CV was considered as **1000 kg**, and the maximum capacity of a smart bin (garbage bin) was considered as **100 kg**.

A detailed discussion of these case studies can be seen in the published paper.

## Conclusion

Waste collection is one of the essential components of waste management process, comprising various interlinked components such as smart bins, dynamic routing, smart collection vehicles, and their coordination. The existing research is either focused on static models or lacks the integration of these components with realistic objectives. This paper, to fill the gaps, implements a flexible real-time route optimization model that accepts and adapts to constantly updating data to provide optimal routes while maximizing the collected waste and minimizing the distance traveled by each CV implemented in an ABM environment. This makes the model suitable for onground implementations as it can take care of unforeseen circumstances and automatically adapt to them. The model was executed for the city of Chandigarh, and it was found that the dynamic routes can **reduce the distance traveled by up to 45%** for the same amount of waste collected using existing static methods. Various execution cases to support the waste collection process in resource constrained societies show the model’s effectiveness in identifying the required resources to satisfy the demand in dynamic environments. 

The outcomes as a planning tool can help make decisions concerning the compromises for limited resources and their impact on waste collection, and extra distance traveled to fulfill the demand. One of the study’s limitations would be the non-consideration of a bin by any other CV, even if the bin were not full when visited. This can be addressed by relaxing the constraint, and its impact on outcomes can be examined. We have considered simulated smart bins for testing models, which can be replaced with IoT-enabled smart bins in real environments. Further integration of real-time data of accidents, construction work, etc., can provide more accurate routes.

## Folder structure and files

  - **Data :** It contains all data which we are using for our computation.
  - **Bin Locations.csv :** They are randomly generated points in QGIS and clustered using K-Means. The depot is assigned ward -1.
  - **distances.csv :** This contains the distance matrix of all points in same ward or from depot.
  - **Static Data :** Contains data regarding Static cases
  - **Dynamic Data :** Contains Data regarding Dynamic cases
  - **Visited Truck # :** Visited truck list with fill ratio
  - **Truck # Data :** Each truck data used for computation.
  - **Statistics.csv :** Stats for the respective case.
  
## Python Files and their usecase
  - **get_distance.py :** Run this to get distance matrix
  - **static_function.py :** Static Optimization code
  - **dynamic_function.py :** Dynamic Optimization code
  - **multi_truck_function.py :** Multiple trucks dynamic optimization function. It works only till 3 trucks in each ward.
  - **four_plus_truck_function.py :** Function to perform dynamic optimzation on 4+ trucks
  - **four_plus_truck_worst_case_function.py :** Function to perform dynamic optimzation on 4+ trucks in worst case
  - **static_unweighted.py :** Static unweighted optimization code
  - **static_find_weights.py :** Finding the wieghts for static case
  - **dynamic_unweighted.py :** Dynamic unweighted optimization code
  - **dynamic_weight_finding.py :** Finding weights for dynamic case. 
  - **dynamic_worst_case.py :** Computing dynamic worst case optimization
  - **dynamic_weighted_multiple_trucks.py :** Dynamic weighted truck optimization where there are 2 trucks in each ward.
  - **dynamic_weighted_three_trucks.py :** Performing Dynamic optimization of 3 trucks in each ward.
  - **dynamic_weighted_four_trucks.py :** Performing Dynamic optimization of 4 trucks in each ward.
  - **dynamic_weighted_multi_best_case.py :** Best collection ratio for dynamic case (5 Trucks)
  - **dynamic_weighted_multi_worst_case.py :** Worst collection ratio for dynamic case (5 Trucks)
  - **show_routes.py :** For visualizing routes.

## Licensing

The following code is open to share, modify and used as a base under the [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/)
