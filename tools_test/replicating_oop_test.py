
#not to be used: "pure" python takes to much time in order to get a suficient number of sample paths
#this is due to the "path dependence" of the computations that didnt allow me to use numpy vectorizable operations to improve the runtimes
#although it could be argued that one could vectorize the operations of the outter loop (the one that tracks the sample paths) , to be tested latter

import numpy as np
import scipy.stats as sp



class Option_fixed_time():
    
    
    def __init__(self,price_0,actual_mean,actual_vol,interest_rate,expire_date,strike_price,transaction_cost,n_steps,n_paths):
        self.price_0=price_0

        self.actual_mean=actual_mean

        self.actual_vol=actual_vol
    
        self.interest_rate=interest_rate
    
        self.expire_date=expire_date
    
        self.strike_price=strike_price

        self.transaction_cost=transaction_cost
    
        self.n_steps=n_steps
    
        self.n_paths=n_paths

        self.delta_t=expire_date/n_steps

    
    def delta(self, s ,t):
        pass

    
    def pricing(self,s,t):
        pass

    
    def exercise(self,s):
        pass


    def replicating_error(self):
    
        

        error=np.empty(self.n_paths)

        compound=np.exp(self.interest_rate*self.delta_t)

        for j in range(0,self.n_paths):
            path=np.empty(self.n_steps+1)
            d=np.empty(self.n_steps)
            q=np.empty(self.n_steps)

            path[0]=self.price_0

            d[0]=self.delta(path[0],0)
            q[0]=self.pricing(path[0],0)-d[0]*path[0]-np.abs(d[0])*self.transaction_cost*path[0]

            for i in range(1,self.n_steps):
                path[i]=path[i-1]*np.exp((self.actual_mean-0.5*self.actual_vol**2)*(self.delta_t)+self.actual_vol*np.sqrt(self.delta_t)*np.random.normal())
                d[i]=self.delta(path[i],self.delta_t*i)
                q[i]=q[i-1]*compound - (d[i]-d[i-1])*path[i] - np.abs(d[i]-d[i-1])*path[i]*self.transaction_cost


            path[self.n_steps]=path[self.n_steps-1]*np.exp((self.actual_mean-0.5*self.actual_vol**2)*(self.delta_t)+self.actual_vol*np.sqrt(self.delta_t)*np.random.normal())
            
            error[j] = (q[self.n_steps-1]*compound+d[self.n_steps-1]*path[self.n_steps]) - self.exercise(path[self.n_steps])


        return error
    
class Short_european_call_bs(Option_fixed_time):
    
    def delta(self, s ,t):
        d1=(np.log(s/self.strike_price)  +      (self.interest_rate + (self.actual_vol**2)/2  ) * (self.expire_date-t) )   /(self.actual_vol*np.sqrt(self.expire_date-t))
        
        return sp.norm.cdf(d1)

    def pricing(self,s,t):
        d1=(np.log(s/self.strike_price)  +      (self.interest_rate + (self.actual_vol**2)/2  ) * (self.expire_date-t) )   /(self.actual_vol*np.sqrt(self.expire_date-t))

        d2=d1-self.actual_vol*np.sqrt(self.expire_date-t)
        
        c0=s*sp.norm.cdf(d1)-self.strike_price*np.exp(-self.interest_rate*(self.expire_date-t))*sp.norm.cdf(d2)

        return c0

    def exercise(self,s):
        return max(s-self.strike_price,0)


class Short_european_call_leland(Option_fixed_time):
    def __init__(self,price_0,actual_mean,actual_vol,interest_rate,expire_date,strike_price,transaction_cost,n_steps,n_paths):
        Option_fixed_time.__init__(self,price_0,actual_mean,actual_vol,interest_rate,expire_date,strike_price,transaction_cost,n_steps,n_paths)
        self.modified_vol=  np.sqrt((self.actual_vol**2 )* (1+(self.transaction_cost/self.actual_vol)*np.sqrt(8/(self.delta_t*np.pi))))

    def delta(self, s ,t):
        d1=(np.log(s/self.strike_price)  +      (self.interest_rate + (self.modified_vol**2)/2  ) * (self.expire_date-t) )   /(self.modified_vol*np.sqrt(self.expire_date-t))
        
        return sp.norm.cdf(d1)

    def pricing(self,s,t):
        d1=(np.log(s/self.strike_price)  +      (self.interest_rate + (self.modified_vol**2)/2  ) * (self.expire_date-t) )   /(self.modified_vol*np.sqrt(self.expire_date-t))

        d2=d1-self.modified_vol*np.sqrt(self.expire_date-t)
        
        c0=s*sp.norm.cdf(d1)-self.strike_price*np.exp(-self.interest_rate*(self.expire_date-t))*sp.norm.cdf(d2)

        return c0

    def exercise(self,s):
        return max(s-self.strike_price,0)


if __name__=="__main__":
    import time


    time_0=time.time()
    option=Short_european_call_bs(price_0=100,actual_mean=0.05,actual_vol=0.25,interest_rate=0.05,expire_date=1,strike_price=100,transaction_cost=0.01,n_steps=100,n_paths=200)
    result=option.replicating_error()

    print(np.average(result))
    print(np.std(result))


    option1=Short_european_call_leland(price_0=100,actual_mean=0.05,actual_vol=0.25,interest_rate=0.05,expire_date=1,strike_price=100,transaction_cost=0.01,n_steps=100,n_paths=200)

    result1=option1.replicating_error()


    print(np.average(result1))
    print(np.std(result1))

    #aproximation of simulating 200000 paths
    print((time.time()-time_0)*1000)