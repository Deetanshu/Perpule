# Perpule Assignment

This assignment pertains to the managing of inventory levels. There are two files, Assignment_run.py and Assignment.ipynb. Assignment.ipynb is to see the process and working of the assignment while the former executes the task in its entirety. 

## Getting Started

In order to run these two, a python 3.6 environment with several prerequisites needs to be activated prior to execution.

### Prerequisites

* pandas
* scikit-learn
* numpy
* math
* argparse

### Running Assignment_run.py

Run Assignment_run.py in the following way if the data files are in the same directory:

```
python Assignment_run.py
```
in the console.
In the event that the data files are located in different directories, make use of arguments to show the path.
```
python Assignment_run.py -c <PATH_TO_CATALOG> -x <PATH_TO_HISTORICAL_DATA> -i <PATH_TO_CURRENT_INVENTORY>
```
Use ```--help``` to see the arguments if there are any doubts.

### Running Assignment.ipynb
In a console window at the same directory, run ```jupyter notebook Assignment.ipynb```
## What does the program do?

The program goes through the following steps:
1. Load all the data.
2. Obtain the list of products.
3. Filter out all data that isn't relevant.
4. Iterate through the products and predict the number of products likely to be moved.
5. Organize data in the format that is requested.
6. Compute answer to Question 2.
7. Save data in the form of a CSV (output.csv)
8. Print answer to Question 2.

### Loading and obtaining "required information"

This is done by:

* The program loads the data using the ```pandas``` library and its ```read_csv``` function.
* The data loaded is stored in a dataframe. 
* The product names are obtained by converting the dataframe into a list. 
* The historical data is filtered by checking to see if the rows have data pertaining to the product list.
* The ```orderid```, ```customerid```, ```barcode``` columns are dropped.
* The inventory is also filtered in the same way as above.

### Predicting the future requirements of products

This is done by predicting the ratio of number of products to the number of transactions.
* The data is first converted into the required format ( a ```numpy``` array).
* The averages and standard deviation of the quantity of products sold in a month and the ratio are computed. This is used to determine the upper limit in the event that the accuracy of the prediction is wildly off.
![equation](https://latex.codecogs.com/gif.latex?L_u_p_p_e_r%20%3D%20%5Cmu_q%20&plus;%202%5Ctimes%20%5Csigma_q)
* A linear feature is computed and fed into a linear regression model using ```scikit-learn```. 
* A 6 month cumulative total is taken and returned.
* The number of days that the current inventory will last is also computed by obtaining the quantity sold per day and then dividing.


## List of things that were tried and did not work.

* Support Vector Regression with a Radial Basis Function kernel. Inconsistent amounts of data meant it worked well for certain cases but not throughout.
* Polynomial regression with >1 degree features - Sometimes the trendline would go steeply downwards and negative values would be predicted. 
* Predicting quantity directly - This didn't work as the fluctuations were very large at times.

## Future improvements:

* With more data (larger time frame), seasonality could be factored in.
* With data such as ordering cost, carrying cost, reorder amounts can be calculated better.
![equation](https://latex.codecogs.com/gif.latex?Reorder%20amount%20%3D%20%5Csqrt%7B%20%5Cfrac%7B2%5Ctimes%20au%5Ctimes%20C_%7Border%7D%7D%7BC_%7Bcarry%7D%7D%7D)


## Authors

Deeptanshu Paul
