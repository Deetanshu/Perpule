# imports
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from math import sqrt, ceil
import argparse

# Constant
DAYSINMONTH=[31,28,31,30,31,30,31,31,30,31,30,31]

# Read all files and remove unnecessary information
def read_all(catalog_dir, historical_dir, inventory_dir):
    catalog = pd.read_csv(catalog_dir)
    historical_all = pd.read_csv(historical_dir)

    product_names = catalog['Product Name'].values.tolist()
    del catalog

    # removing irrelevant data from historical dataframe.
    condition = (historical_all['productname'].isin(product_names))
    historical = historical_all.loc[condition]
    
    del historical_all
    historical = historical.drop(['orderid', 'customerid', 'barcode'], axis=1)

    historical['ordertime'] = pd.to_datetime(historical['ordertime'])

    # reading Inventory
    inventory_all = pd.read_csv(inventory_dir)

    # removing irrelevant data from inventory
    condition = (inventory_all['Product Name'].isin(product_names))
    inventory = inventory_all.loc[condition]
    #del inventory_all,condition

    # resetting index for all
    historical = historical.reset_index()
    inventory = dict(inventory.values.tolist())

    return historical, inventory, product_names

# Polynomial regression model and prediction. This did not work.
def poly_regr(info, curr_inv,  X_feat = 'timestamp', y_feat = 'qi', qi = True):

    # Create X and y to be fed into model.
    X = []
    y = []
    temp = 0
    quant = 0
    for x in info:
        X.append(int(x[X_feat]))
        y.append(x[y_feat])
        temp=temp+x['count']
        quant = quant + x['quantity']
    X = np.array(X).reshape(-1,1)
    y = np.array(y)
    if len(X) == 0:
        return 0, None, None
    avg_y = sum(y)/len(X)
    avg_q = quant/len(X)
    stddev_y = 0
    stddev_q = 0
    for i in range(len(X)):
        stddev_y = stddev_y + abs(avg_y - y[i])
        stddev_q = stddev_q + abs(avg_q - info[i]['quantity'])
        
    temp = temp/len(X)
    one = ceil((avg_q+2*stddev_q)*6)
    two = (avg_y+2*stddev_q)
    # Create features
    poly = PolynomialFeatures(degree = 1)
    poly_x = poly.fit_transform(X)
    
    # Create model and train
    model = LinearRegression()
    model.fit(poly_x, y)
    total = 0
    qi_avg = 0
    for i in range(1,6):
        if not qi:
            total = total + model.predict(poly.fit_transform(np.array([int(pd.datetime(2019,i,1).timestamp())]).reshape(-1, 1)))[0]
        else:
            qi_pred = max(min(model.predict(poly.fit_transform(np.array([int(pd.datetime(2019,i,1).timestamp())]).reshape(-1, 1)))[0],two),1)
            total = total + qi_pred*temp
            qi_avg += qi_pred
    total = min(ceil(total),one)
    days = int(curr_inv*181/total)
    return total, model, days

# Process prediction for ONE product: (Note that df is going to be "historical")
def pred_six_months(productname, df, curr_inv):

    # narrow down information to what's necessary.
    condition = (df['productname']==productname)
    df = df.loc[condition].reset_index()

    # get raw information from raw data.
    info = []
    size = len(df)
    for i in range(size):
        curr_dt = df.ordertime.iloc[i]
        year = curr_dt.year + int((curr_dt.month+1)/12)
        month = max((curr_dt.month + 1) % 13, 1)
        day = min(DAYSINMONTH[month-1], curr_dt.day)
        up_limit = pd.datetime(year, month, day)
        if up_limit >= df.ordertime.iloc[size-1]:
            break
        
        j = i
        total_quantity = 0
        count = 0
        while df.ordertime[j] < up_limit:
            total_quantity = total_quantity + df.qty.iloc[j]
            j = j+1
            count = count + 1
        info.append({"date":curr_dt,
         "timestamp":curr_dt.timestamp(),
         "quantity":total_quantity,
         "count":count,
         "qi":(total_quantity/count)})
    
    # Feed into polynomial regression model and return value
    #pred,model,poly = 
    return poly_regr(info, curr_inv)
    

# Gets days before stock runs out - Not working and not used.
def get_days(model, curr_inv, pred, poly):
    posneg = None
    if pred < curr_inv:
        return 181
    est = int((181/pred)*curr_inv)
    p=None
    #print("Originial EST: ", est)
    while True:
        x = est
        if est < 0:
            return int((181/pred)*curr_inv)
        for i in range(12):
            x = x - DAYSINMONTH[i]
            if x < 0:
                x = i + 1
                break
            p = x if x > 0 and x <= DAYSINMONTH[i] else 1
        if p is None:
            p = x if x > 0 and x <= DAYSINMONTH[i] else 1
        xyz = int(model.predict(poly.fit_transform(np.array([int(pd.datetime(2019,x,p).timestamp())]).reshape(-1, 1))))
        #print("Calc is ", xyz, " curr inv is ", curr_inv)
        if xyz == curr_inv:
            return est
        elif xyz > curr_inv:
            if posneg is None:
                posneg = False
            elif posneg:
                return est 
        elif xyz < curr_inv:
            if posneg is None:
                posneg = True
            elif not posneg:
                return est + 1
        
        if posneg:
            est = est - 1
        else:
            est = est + 1
        #print("EST = ", est)

def get_reorder_amt(pred):
    return sqrt(4*pred*1.5)

#Prints progress bar
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '#'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()
        
def orchestrate(historical, inventory, product_names):
    data = []
    q2_p1 = False
    leftover = 0
    itr = 0
    for product_name in product_names:
        curr_inv = inventory[product_name]
        pred, model, days = pred_six_months(product_name, historical, curr_inv)
        if pred == 0:
            data.append([product_name, curr_inv, 0, 0])
            continue
        #DEBUG
        #print(product_name, "\t", pred)
        printProgressBar(itr,len(product_names))
        #days = get_days(model, curr_inv, pred, poly)
        data.append([product_name, curr_inv, days, int(max(pred-curr_inv, 0))])
        if product_name == 'Britannia Bisc Gday Cashew 100g':
            #num = get_reorder_amt(pred)
            #print("num is ", num, "predicted amount is ", pred)
            if pred > 200:
                q2_p1 = True
                leftover = 300-pred
        del model
        itr = itr+1
    df = pd.DataFrame(data,columns=['Product Name','Current Inventory', "Number of days the current inventory will last starting 1st Jan 2019", "Quantity  to purchase for the next six months"])
    
    #print(df)
    df.to_csv(r'output.csv')
    printProgressBar(100,100)
    print("\n\nAnswer to Q1 has been saved as output.csv\n\n")
    print("The answer to Q2: ")
    if q2_p1:
        print('Yes, the offer is profitable and the total predicted expired units are ',leftover)
    else:
        print("No, the offer isn't profitable. ")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--catalog", default = "britannia_catalogue.csv", help = "Path to catalog file.")
    parser.add_argument("-x", "--historical_dir", default = "historicaldata.csv", help = "Path to historical data file.")
    parser.add_argument("-i", "--inventory", default = "current_inventory.csv", help = "Path to inventory file.")
    args = parser.parse_args()    
    historical, inventory, product_names = read_all(catalog_dir = args.catalog, historical_dir = args.historical_dir, inventory_dir = args.inventory)
    df = orchestrate(historical, inventory, product_names)

