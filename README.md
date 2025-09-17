# Delta-Hedging-Simulator
A Python script that simulates the dynamic delta hedging of a long non-dividend paying European call option. 

This project investigates the P&L of a hedging strategy and how it is driven by the difference between the implied volatility, used for option pricing and hedging, and the realised volatility of the simulated path.

In this project we will be exploring the applications of a form of delta hedging wherein we maintain a delta neutral portfolio in order to isolate the affect volatility has on PnL. In this way we are trading volatility rather than stock movement. 

Firstly, I must introduce the Black-Scholes-Merton model used in this project and the meaning of delta. 

Delta(Δ) measures the sensitivity of the option price to the underlying stock price. Mathematically, it's the partial derivative of option value with respect to stock price. For a call option, Δ ranges from 0 to 1; for a put, from -1 to 0. In Black-Scholes, delta also has a probabilistic interpretation: it can be seen as the approximate probability that the option finishes in the money(eg. -0.25 for put and 0.25 for call delta corresponds to a 25% chance of the option finishing in the money). For call options, delta increases with a stock price increases, reflecting higher probability of the call expiring in the money. For put options, delta decreases with an increase in stock price, reflecting the lower probability of the put expiring in the money. This projects focus is on delta hedging a call European option, a type of option which can only be executed at maturity, but the code can be adapted for a European put option instead.

The model used to price the call and calculate deltas is the Black-Scholes-Merton model. This model assumes that stock price, $S_t$ follows a geometric Brownian motion, meaning returns are normally distributed and prices evolve continuously with both a deterministic drift term $\mu S_t dt$ and a stochastic term $\sigma S_t dW_t$ driven by Weiner process $W_t$. $\sigma$ is the volatility of the stock price:

$$
dS_t= \mu S_t dt+ \sigma S_t dW_t
$$

By defining function F(S,t)=ln(S), we can apply Ito's Lemma and after integrating over time from 0 to t we come to this equation:

$$
\ln\left(\frac{S_t}{S_0}\right) = \left(\mu - \tfrac{1}{2}\sigma^2\right)t + \sigma W_t
$$

From this, it is clear that due to the derministic dt term and by the fact that $\sigma dW_t$ has a mean of 0 and a variance of $\sigma^2 T$, $\ln\left(\frac{S_t}{S_0}\right)$ follows a normal distribution with mean $(\mu - 0.5 \sigma^2)T$ and a variance of $\sigma^2 T$. We say that stock price has a lognormal distribution because of this. We then exponentiate each side of this equation to get the closed form solution for stock price evolution:

$$
S_t = S_0 e^{ (\mu - \frac{1}{2}\sigma^2) t + \sigma W_t }
$$

The derivation above describes the real world evolution of stock price as characterised by the drift term $\mu$. When pricing deivatives such as European call options, we change the measure to the risk-neutral measure, Q. In this measure, the Black-Scholes assumption of no-arbitrage is satisfied, allowing us to price derivatives correctly. We will next discuss how path generation is carried out and how call price and delta are calculated in the risk-neutral measure.

To find a continuous time evolution of stock price $S_t$ under the risk-neutral measure, we need to go back to the real-world $dS_t$ equation and replace $dW_t$ with a new brownian motion $\widetilde{W}_t$ with relation to $dW_t$ described by:

$$
d\widetilde{W}_t = dW_t^{\mathbb{P}} + \frac{\mu - r}{\sigma}  dt
$$

Where r is the risk-free rate.

This gives us a stochastic process for $dS_t$ in the measure Q:

$$
dS_t = r S_t  dt + \sigma S_t  d\widetilde{W}_t
$$

Which is solved in the same way as before. We achieve the closed form solution for stock evolution $S_t$ under Q:

$$
S_t = S_0 e^{(r - \frac{1}{2}\sigma^2)t + \sigma \widetilde{W}_t}
$$

This continuous time evolution can be adapted to a discrete time step evolution which we use to generate the stock paths in the script. We take discrete timesteps dt to calculate the next stock price movement- in the code, dt only refers to this discrete timestep. To adapt this equation, we first recognise that $dW_t$ has mean 0, variance dt and a normal distribution. To replace this term we use a standard normal random variable Z which is normally distributed with mean 0 and variance 1 and multiply it by $\sqrt(dt)$ so the variance becomes dt. Over num_steps(number of timesteps) iterations we can generate our stock path from seed $S_0$:

     for t in range(1, num_steps+1):
            
            paths[t]=paths[t-1]*np.exp((r-0.5*sigma_real**2)*dt+sigma_real*Z[t-1]*np.sqrt(dt))

Before this loop, we must define variables num_steps and num_paths simulate multiple stock price evolutions over multiple timesteps. After defining dt=T/num_steps we create a matrix of zeros with dimensions (num_steps+1, num_paths): 

paths=np.zeros((num_steps+1, num_paths))

We add an extra row so we can initialise the zeroth row with $S_0$ and so the remaining rows enumerate the timesteps. 

You might notice that sigma_real was used in calculating each movement and not simply named 'sigma'. The aim of this project is to isolate and investigate the affect volatility has on the average PnL over all paths, and to do so we simulate stock movements with sigma_real or realised volatility and use sigma_imp or implied volatility to calculate current call price and the delta of $S_t$ at each timestep. In short, implied volatility is calculated at t=0 and measures the expected future volatility of the underlying stock at T, whereas realised volatility is the actual volatility the stock price experiences. With a sufficiently large number of paths generated over a sufficiently large number of timesteps(for accurate average PnL's), the trading strategy used in this project makes profit for sigma_imp<sigma_real and a loss for sigma_imp>sigma_real. In this way we are long on volatility and are betting on realised volatility beating market expectation.

Under Q, the price of a derivative is its discounted expected payoff. For a European call:

$$
c_0 = e^{-rT} \cdot \mathbb{E}^{\mathbb{Q}}\left[ \max(S_T - K, 0) \right]
$$

This pricing ensures no arbitrage in a complete market to fit the criteria for the the Black-Scholes models assumptions; If we buy a call at T=0 we have $-c_0$ in our cash account which grows to $- \mathbb{E}^{\mathbb{Q}}\left[ \max(S_T - K, 0) \right]$ at maturity. Since this is the negative of our expected payoff, the total profit a trader can expect from buying a call is exactly 0, thus no arbitrage opportunities.

There are two main methods to find the closed form solution for c; by solving the Black-Scholes PDE or the risk-neutral approach. The latter is what we will use since we have not yet derived the PDE and have the correct setup to solve using expectations. 

We can write the payoff of a call option as

$$
\max(S_T - K, 0) = (S_T - K)\mathbf{1}_{\{S_T > K\}}.
$$

Taking expectations under the risk-neutral measure $\mathbb{Q}$ gives

$$
c_0 = e^{-rT} \mathbb{E}^\mathbb{Q}\left[ (S_T - K)\mathbf{1}_{\{S_T > K\}} \right].
$$

We split this into two terms:

$$
c_0 = e^{-rT} \Big( \mathbb{E}^\mathbb{Q}[ S_T \mathbf{1}_{\{S_T > K\}} ] - K \mathbb{E}^\mathbb{Q}[ \mathbf{1}_{\{S_T > K\}} ] \Big).
$$

To evaluate the second term, notice that

$$
\mathbb{E}^\mathbb{Q}[ \mathbf{1}_{\{S_T > K\}} ] = \mathbb{Q}(S_T > K).
$$

Using the closed-form solution for $S_T$ under $\mathbb{Q}$,

$$
S_T = S_0 \exp\Big( (r - \tfrac{1}{2}\sigma^2)T + \sigma \sqrt{T} Z \Big), 
\quad Z \sim \mathcal{N}(0,1).
$$

The inequality $S_T > K$ becomes

$$
Z > \frac{\ln(K/S_0) - (r - \tfrac{1}{2}\sigma^2)T}{\sigma \sqrt{T}}.
$$

Define

$$
d_2 = \frac{\ln(S_0/K) + (r - \tfrac{1}{2}\sigma^2)T}{\sigma \sqrt{T}}.
$$

Then the inequality is equivalent to $Z > -d_2$, so

$$
\mathbb{Q}(S_T > K) = \mathbb{Q}(Z < d_2) = N(d_2),
$$

where $N(\cdot)$ is the CDF of a standard normal random variable.

Thus, the second term inside the bracket of the call price becomes

$$
-K N(d_2).
$$

For the first term we perform a change of measure. Define the stock measure $(\widetilde{\mathbb Q}\)$ by the Radon-Nikodym derivative

$$
\frac{d\widetilde{\mathbb Q}}{d\mathbb Q}
= \frac{S_T}{S_0 e^{rT}}.
$$

Then

$$
\mathbb{E}^{\mathbb Q}[S_T\mathbf{1}_{\{S_T>K\}}]
= S_0 e^{rT}\widetilde{\mathbb E}\big[\mathbf{1}_{\{S_T>K\}}\big]
= S_0 e^{rT}\widetilde{\mathbb Q}(S_T>K).
$$

Under $\mathbb{Q}$ we may write $S_T$ as

$$
S_T = S_0\exp\Big((r-\tfrac12\sigma^2)T + \sigma\sqrt{T} Z\Big)
$$

Under the measure $\widetilde{\mathbb{Q}}$, the drift of $\ln S_T$ increases by $\sigma^2 T$ compared to $\mathbb{Q}$. Therefore, $S_T$ becomes:

$$
S_T = S_0 \exp\left( (r + \tfrac{1}{2}\sigma^2)T + \sigma \sqrt{T} Z \right),
$$ 

To compute $\widetilde{\mathbb Q}(S_T>K)\$ we solve the inequality similarly to before:

$$
S_T > K
\quad\Longleftrightarrow\quad
\sigma\sqrt{T} Z > \ln(K/S_0) - (r+\tfrac12\sigma^2)T
$$

$$
\Longleftrightarrow\quad
Z > \frac{\ln(K/S_0) - (r+\tfrac12\sigma^2)T}{\sigma\sqrt{T}} = -d_1,
$$

where

$$
d_1 = \frac{\ln(S_0/K) + (r+\tfrac12\sigma^2)T}{\sigma\sqrt{T}}.
$$

So

$$
\widetilde{\mathbb Q}(S_T>K) = \widetilde{\mathbb{Q}}(Z > -d_1) \=\ \widetilde{\mathbb{Q}}(Z < d_1)=N(d_1),
$$

and therefore the first term is

$$
\mathbb{E}^{\mathbb Q}[S_T\mathbf{1}_{\{S_T>K\}}] = S_0 e^{rT} N(d_1).
$$

Putting both terms together and discounting by $e^{-rT}$ yields the Black-Scholes call price:

$$
c_0 = e^{-rT}\Big(S_0 e^{rT} N(d_1) - K N(d_2)\Big) = S_0 N(d_1) - K e^{-rT} N(d_2).
$$

If we want to find the price of a non-dividend paying European put, we can use the put-call parity:

$$
p_0 = c_0 + K e^{-rT} - S_0 = S_0 N(d_1) - K e^{-rT} N(d_2) + K e^{-rT} - S_0.
$$

Simplifying, we obtain:

$$
p_0 = K e^{-rT} \big(1 - N(d_2)\big) - S_0 \big(1 - N(d_1)\big) = K e^{-rT} N(-d_2) - S_0 N(-d_1),
$$

since $(1 - N(x)) = N(-x)$ for the standard normal distribution.

Now that we have an expression for $c_0$, we can calculate $\Delta$ using the partial derivative:

$$
\Delta = \frac{\partial c_0}{\partial S_0} = \frac{\partial}{\partial S_0} \left( S_0 N(d_1) - K e^{-rT} N(d_2) \right).
$$

Recall that $d_1$ and $d_2$ are functions of $S_0$:

$$
d_1 = \frac{\ln(S_0/K) + (r + \tfrac{1}{2}\sigma^2)T}{\sigma \sqrt{T}}, \quad
d_2 = d_1 - \sigma \sqrt{T}.
$$

Applying the product rule and chain rule:

$$
\Delta = N(d_1) + S_0 \frac{\partial N(d_1)}{\partial S_0} - K e^{-rT} \frac{\partial N(d_2)}{\partial S_0}.
$$

Note that $\frac{\partial N(d_1)}{\partial S_0} = n(d_1) \frac{\partial d_1}{\partial S_0}$ and $\frac{\partial N(d_2)}{\partial S_0} = n(d_2) \frac{\partial d_2}{\partial S_0}$, where $n(x)$ is the standard normal density function. Compute the derivatives:

$$
\frac{\partial d_1}{\partial S_0} = \frac{1}{S_0 \sigma \sqrt{T}}, \quad
\frac{\partial d_2}{\partial S_0} = \frac{1}{S_0 \sigma \sqrt{T}}.
$$

Substitute:

$$
\Delta = N(d_1) + S_0 n(d_1) \cdot \frac{1}{S_0 \sigma \sqrt{T}} - K e^{-rT} n(d_2) \cdot \frac{1}{S_0 \sigma \sqrt{T}}.
$$

Simplify:

$$
\Delta = N(d_1) + \frac{1}{\sigma \sqrt{T}} \left( n(d_1) - \frac{K e^{-rT}}{S_0} n(d_2) \right).
$$

It can be shown that $S_0 n(d_1) = K e^{-rT} n(d_2)$. To see this, note:

$$
n(d_1) = \frac{1}{\sqrt{2\pi}} e^{-d_1^2/2}, \quad
n(d_2) = \frac{1}{\sqrt{2\pi}} e^{-d_2^2/2}.
$$

Since $d_2 = d_1 - \sigma \sqrt{T}$, we have:

$$
d_2^2 = d_1^2 - 2 d_1 \sigma \sqrt{T} + \sigma^2 T.
$$

Then,

$$
n(d_2) = n(d_1) \exp\left( d_1 \sigma \sqrt{T} - \tfrac{1}{2} \sigma^2 T \right).
$$

But from the definition of $d_1$:

$$
d_1 \sigma \sqrt{T} = \ln(S_0/K) + (r + \tfrac{1}{2}\sigma^2)T
$$

so:

$$
n(d_2) = n(d_1) \exp\left( \ln(S_0/K) + rT \right) = n(d_1) \cdot \frac{S_0}{K} e^{rT}.
$$

Therefore,

$$
K e^{-rT} n(d_2) = S_0 n(d_1).
$$

Substituting back, the terms in brackets cancel:

$$
\Delta = N(d_1) + \frac{1}{\sigma \sqrt{T}} \left( n(d_1) - n(d_1) \right) = N(d_1).
$$

Thus, the Delta of a European call option is:

$$
\Delta = N(d_1).
$$

We now have all the right ingredients for our trading strategy. My chosen method of Delta-hedging is to buy a non-dividend paying European call option priced c_0, and borrow $\Delta_0 S_0$ worth of stock and instantly sell it, also known as shorting. The portfolio value at time 0 is $\Pi=c_0- \Delta_0 S_0$ and our cash account has value $cash= -c_0+\Delta_0 S_0$. At each timestep, the only action the trader takes is to rebalance the portfolio so that the short position is $\Delta_0 S_0$. This keeps the portfolio Delta-neutral, meaning our PnL is indifferent to stock price movements. Therefore our PnL is solely dependant on volatility beating market expectation to make a profit. The implied volatility is priced into the call bought at t=0, so if realised volatility is greater than implied volatility, we had bought $c_0$ at a premium. To rebalance, we sell stock if there's an increase in stock price and buy for a decrease in stock price, since Delta moves with stock price and we have a negative position of shares. We benefit from large swings in stock price. 

This loop describes the hedging strategy, along with the cash accrued with a risk-free growth rate r:

     def delta_calculator(S,K,Tau,r,sigma_imp):
            d1=((np.log(S/K)+(r+0.5*sigma_imp**2)*Tau)/(sigma_imp*np.sqrt(Tau)))
            delta=norm.cdf(d1)
            return delta
    
    deltas = np.zeros_like(paths)
    
    for i in range(0, num_steps):
        deltas[i]=delta_calculator(paths[i],K,T-i*dt,r,sigma_imp)
    deltas[-1]= (paths[-1]>K).astype(float)
    
    call_price = S0*norm.cdf((np.log(S0/K)+(r+0.5*sigma_imp**2)*T)/(sigma_imp*np.sqrt(T)))-K*np.exp(-r*T)*norm.cdf(((np.log(S0/K)+(r-0.5*sigma_imp**2)*T)/(sigma_imp*np.sqrt(T))))
    cash_flow = np.zeros(num_paths)
    cash_flow[:] = paths[0]*delta0 - call_price
    
    for n in range(1, num_steps+1):
        hedge = paths[n]*(deltas[n]-deltas[n-1])
        cash_flow *=exp_rdt
        cash_flow += hedge

We already have a matrix of stock prices at every time-step for each path so calculating the corresponding Deltas requires a calculator and a loop of its own as shown above. Since the call is either 100% in-the-money or 100% out-the-money at t=T, $\Delta$ is either 0 or 1. But we run into a problem since at t=T, $Tau = T-i dt$ so an error occurs if we use the calculator for the final row since we end up dividing by 0 to calculate d1, so we use the line deltas[-1]= (paths[-1]>K).astype(float) instead.

We then initialise the cash account, called cash_flow here, with the initial hedge, paths[0]* delta0 minus the call price accross all paths and before hedging, we multiply cash_flow by the risk-free growth for time dt: exp_rdt=$\exp(r dt)$ defined earlier in the code (to avoid unnecessary recomputation). We then add paths[n]*(deltas[n]-deltas[n-1]) to the cash account which corresponds to buying or selling the correct amount of stock to rebalance the portfolio.

We do these calculations for an array of different realised volatilities:

      realised_vols=[.1,.15,.2,.23,.26,.31,.36,.41,.46,.51]

We keep the following parameters constant: S0=238, sigma_imp=0.23 based on the current market data of an Apple stock. Choose T=1, so our call expires 1 year from the time bought. Finally, we use r=0.0385 which is the yield for a 1 year U.S Treasury bond. We may vary K to test the accuracy of the code in measuring volatility dependant average PnL's. 

To calculate PnL per path, We consider the final amount in the cash account after hedging, excecuting the call if in the money, and finally closing our short position. After hedging is complete, we add money gained from trading the volatility of stock (+cash_flow). We then excecute the call if $\Delta_T=1$ ($+(S_T-K)$) and we close the postion by buying back shares shorted (-$\Delta_T S_T$). 

    exercise=np.maximum(paths[-1]-K, 0)
    PnL = cash_flow -deltas[len(paths)-1]*paths[len(paths)-1]+exercise

For each realised_vols: we calculate the mean PnL, standard deviation and sampling error of PnL's across all paths:

    mean_PnLs=[] #initialise before the for loop
    std_PnLs=[]
    se_PnLs=[]

     exercise=np.maximum(paths[-1]-K, 0) 
    PnL = cash_flow -deltas[len(paths)-1]*paths[len(paths)-1]+exercise
    mean_PnLs.append(PnL.mean())
    std_PnLs.append(PnL.std(ddof=1))
    se_PnLs.append(PnL.std(ddof=1)/np.sqrt(num_paths))

PnL.mean() automatically calculates the mean PnL and std_PnLs.append(PnL.std(ddof=1)) calculates the standard deviation of PnLs across all paths for a given realised volatility value. se_PnLs is the sampling error which is calculated by dividing the standard deviation of path PnL's by the square root of the number of paths ($\frac{\sigma}{\sqrt{n}}$). If we have a large standard deviation for a small sample size of PnL's, then we can increase num_paths for a smaller sampling error, thus a greater certainty in our calculation for mean PnL's. It's also optimal to increase num_steps as much as possible to replicate continuous hedging as closely as possible. There is a theoretical maximum possible profit for a minimum hedging error for a given path due to delta hedging and this limit is reached as $\lim_{\Delta t \to 0}$.

For better analysis of results, we fix a matrix of randomly generated numbers on a standard normal distribution before the loop:

       rng = np.random.default_rng(0)
       Z = rng.standard_normal(size=(num_steps, num_paths))

This ensures that across the sigma_real's, the "paths" matricies have the same stochastic path movements, just scaled differently by sigma_real, ensuring that the differences in PnL mean and PnL SD measurements are only dependant on volatility and are not influenced by noise. Noise in this context refers to the randomness of each Z[i] shock which contributes to sampling variation outside of volatility. By fixing Z across runs, we can isolate the effect of volatility rather than conflating it with sampling variation.

## Results

Before analysing results, we need to ensure they are both accurate and representative. The two main sources of error that can distort our findings are hedging error and sampling error.

Hedging error arises because delta hedging is done at discrete time steps rather than continuously. Increasing num_steps reduces this error and brings the simulation closer to the theoretical limit of continuous hedging. Sampling error arises because we only simuate a finite number of paths. Increasing num_paths enlarges the sample set, reducing the standard error of the estimated mean and standard deviation of PnL for each realised volatility.

We can only add so many paths and steps, until the code becomes too computationally costly to run. So we must strike a balance between computational cost with accuracy. 

My desired maximum sampling error across runs is $SE_{\max}$ = 0.05. The largest PnL standard deviation, $sigma_{\max}$ is roughly 14.7 for $sigma_{\real} = 0.51$ so rearranging the equation for sampling error, $SE = \frac{\sigma}{\sqrt{N}}$, we find the minimum required paths:
$$
N_{\min} = (\frac{14.7}{0.05})^2 = 294^2 = 86436
$$


