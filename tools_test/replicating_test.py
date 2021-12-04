from numba.np.ufunc import parallel
import numpy as np
from numba.experimental import jitclass
from numba import njit ,prange
from numba import double,int32
import matplotlib.pyplot as plt
from math import erfc



#stript used for testing the performance of some of the features of numba

@njit(fastmath=False)
def normcdf(x):
    return erfc(-x / np.sqrt(2.0)) / 2.0

@njit(fastmath=False)
def delta(s,T,t,sigma,r,k):
    d1=(np.log(s/k)  +      (r + (sigma**2)/2  ) * (T-t) )   /     (sigma*np.sqrt(T-t))
    
    return normcdf(d1)

@njit(fastmath=False)
def exercise(s,k):
    return max(s-k,0)


@njit(fastmath=False)
def bs_pricing(s,T,sigma,r,k):
    
    d1=(np.log(s/k)+(r+(sigma**2)/2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    c0=s*normcdf(d1)-k*np.exp(-r*T)*normcdf(d2)
    return c0

@njit(fastmath=False,parallel=False)
def replicating_error_black_sholes(price_0,actual_mean,actual_vol,interest_rate,expire_date,strike_price,transaction_cost,n_steps,n_paths):
        
        delta_t=expire_date/n_steps
        #hedging error 
        error=np.empty(n_paths)
        #1+interest in delta_t time interval
        compound=np.exp(interest_rate*delta_t)

        for j in prange(0,n_paths):
            #this values are kept in an array in order to make debbuing easier  
            path=np.empty(n_steps+1)
            d=np.empty(n_steps)
            q=np.empty(n_steps)

            path[0]=price_0
            #quantity of stocks- equal to the delta of the option 
            d[0]=delta(path[0],expire_date,0,actual_vol,interest_rate,strike_price)
            #rf bank account at t=0- revenue of selling the option  
            q[0]=bs_pricing(path[0],expire_date,actual_vol,interest_rate,strike_price)-d[0]*path[0]-np.abs(d[0])*path[0]*transaction_cost
            
            
            for i in range(1,n_steps):
                #genarate next price using the gbm formula
                path[i]=path[i-1]*np.exp(    (actual_mean-0.5*actual_vol**2)  *delta_t       +     actual_vol*np.sqrt(delta_t)*np.random.normal())
                #quantity of stocks- equal to the delta of the option 
                d[i]=delta(path[i],expire_date,i*delta_t,actual_vol,interest_rate,strike_price)
                #rf bank account- (last value+interest) + cost/revenue of buying stocks + TC of buying stocks 
                q[i]=q[i-1]*compound - (d[i]-d[i-1])*path[i] - np.abs(d[i]-d[i-1])*path[i]*transaction_cost

            #genarate final price of the stock    
            path[n_steps]=path[n_steps-1]*np.exp((actual_mean-0.5*actual_vol**2)*delta_t+actual_vol*np.sqrt(delta_t)*np.random.normal())
            
            #Any of the papers described how transaction cost are computed at expire date, so I've assumed by now that arent TC at expire for now
            #hedgin error=last value in bank account + interest + last quantity of stocks* new price - payoff of the option
            error[j] = q[n_steps-1]*compound+d[n_steps-1]*path[n_steps] - exercise(path[n_steps],strike_price)

        return error
@njit(fastmath=True,parallel=False)
def replicating_error_leland(price_0,actual_mean,actual_vol,interest_rate,expire_date,strike_price,transaction_cost,n_steps,n_paths):
        


        delta_t=expire_date/n_steps

        modified_vol=  actual_vol *np.sqrt(  1    +    (transaction_cost/actual_vol)    *  np.sqrt(   8    /    (delta_t*np.pi))   )

        #hedging error 
        error=np.empty(n_paths)
        #1+interest in delta_t time interval
        compound=np.exp(interest_rate*delta_t)

        for j in prange(0,n_paths):
            #this values are kept in an array in order to make debbuing easier  
            path=np.empty(n_steps+1)
            d=np.empty(n_steps)
            q=np.empty(n_steps)

            path[0]=price_0
            #quantity of stocks- equal to the delta of the option 
            d[0]=delta(path[0],expire_date,0,modified_vol,interest_rate,strike_price)
            #rf bank account at t=0- revenue of selling the option  
            q[0]=bs_pricing(path[0],expire_date,actual_vol,interest_rate,strike_price)-d[0]*path[0]-np.abs(d[0])*path[0]*transaction_cost
            
            
            for i in range(1,n_steps):
                #genarate next price using the gbm formula
                path[i]=path[i-1]*np.exp(    (actual_mean-0.5*actual_vol**2)  *delta_t       +     actual_vol*np.sqrt(delta_t)*np.random.normal())
                #quantity of stocks- equal to the delta of the option 
                d[i]=delta(path[i],expire_date,i*delta_t,modified_vol,interest_rate,strike_price)
                #rf bank account- (last value+interest) + cost/revenue of buying stocks + TC of buying stocks 
                q[i]=q[i-1]*compound - (d[i]-d[i-1])*path[i] - np.abs(d[i]-d[i-1])*path[i]*transaction_cost

            #genarate final price of the stock    
            path[n_steps]=path[n_steps-1]*np.exp((actual_mean-0.5*actual_vol**2)*delta_t+actual_vol*np.sqrt(delta_t)*np.random.normal())
            
            #Any of the papers described how transaction cost are computed at expire date, so I've assumed by now that arent TC at expire for now
            #hedgin error=last value in bank account + interest + last quantity of stocks* new price - payoff of the option
            error[j] = q[n_steps-1]*compound+d[n_steps-1]*path[n_steps] - exercise(path[n_steps],strike_price)

        return error


@njit(fastmath=True,parallel=False)
def aux():
    list=np.empty(40)
    list_sd=np.empty(40)

    list_1=np.empty(40)
    list_1_sd=np.empty(40)

    for i in prange(1,41):

        delta_t=i*5

        option=replicating_error_black_sholes(price_0=100,actual_mean=0.05,actual_vol=0.25,interest_rate=0.05,expire_date=1,strike_price=100,transaction_cost=0.01,n_steps=delta_t,n_paths=200000)
        option1=replicating_error_leland(price_0=100,actual_mean=0.05,actual_vol=0.25,interest_rate=0.05,expire_date=1,strike_price=100,transaction_cost=0.01,n_steps=delta_t,n_paths=200000)




        list[i-1]=(np.mean(option))
        list_sd[i-1]=(np.std(option))
        list_1[i-1]=(np.mean(option1))
        list_1_sd[i-1]=(np.std(option1))
        print(delta_t)

    print(list)
    print(list_sd)


    print(list_1)
    print(list_1_sd)

    return list, list_1


if __name__=="__main__":
    import time

    time0=time.time()
    option=replicating_error_black_sholes(price_0=100,actual_mean=0.05,actual_vol=0.25,interest_rate=0.05,expire_date=1,strike_price=100,transaction_cost=0.01,n_steps=100,n_paths=200000)

    print(np.mean(option))
    print(np.std(option))



    option1=replicating_error_leland(price_0=100,actual_mean=0.05,actual_vol=0.25,interest_rate=0.05,expire_date=1,strike_price=100,transaction_cost=0.01,n_steps=100,n_paths=200000)



    print(np.mean(option1))
    print(np.std(option1))

    print(time.time()-time0)

    time0=time.time()
    #a,b=aux()
    print(time.time()-time0)

'''
conclusions:- the oop version is slightly slower to the functional version (9s vs 7.5s for 2 options, not sufficiently faster to justify the loss in abstraction when using this approach)
            - using parallel loops inside the replicating function does not yield significant speedups (7.5s vs 7.2s for 2 options)
            - using ffast math compiler flag (breaks strict IEEE compliance and SIMD support) in the replicating functions does not yield significant speedups (7.5s vs 7.2s for 2 options)
            - using a parallel loop when estimating a batch of options yield acceptable (and significant) speedups, (+/- 200s no parallell vs 70s parallel vs 88s for oop w/parallel for 40 options) although not near the expected "perfect speedup" of 8 due to poor load balancing 

'''