# SynEvaRecSimulator

<h1 align="center"> Performance Ranking of Recommender Systems on Simulated Data
<h3 align="center">This work is an addition to the paper "Performance Ranking of Recommender Systems on Simulated Data" by Stavinova et al.</h3>

## Abstract
Recent studies have shown that modelling and simulation of interactions between a recommender system (RS) and its users have a great potential for accelerating the research and industrial deployment of RSs. Frameworks providing such simulations are called simulators and are widely used for RSs of different types. 
  
Nevertheless, there exist the problem of simulation validation and of the inconsistency of RS performance ranking on real-world and the corresponding synthetic data. 
In this paper, using and extending the recently developed SynEvaRec simulator we propose a validation procedure for simulators of this type and study the consistency of RS performance ranking on response matrices of different sparsity. 
  
It is observed in our experiments that (i) the procedure is an effective tool to see what one may expect from the simulation on real-world data, (ii) the consistency  of RS performance ranking depend on the data considered and even the sample size used for RS training.

## Data
We use two types of datasets in the study:
 - Validation data. The Validation dataset contains 1,000 users and 100 items (both with attributes), and 100,000 user-item responses (i.e. the data is complete). 
   Each user and each item has three attributes, all of them are numeric.
 - The real-world data. The dataset is a small one containing data about Restaurants and Their Clients (available <a href='https://www.kaggle.com/uciml/restaurant-data-with-consumer-ratings'>here</a> ).

The validation data can be generated in our code using the "get_validation_data" function with default parameters. This function is located on
  
```
./AutoRec/load_data
```
## Experiments
All experiments are available in their respective folders:
   - For Validation dataset in folder ./notebooks/validation/
   - For Restaurants dataset - ./notebooks/rests/
