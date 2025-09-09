# Delta-Hedging-Simulator
A Python script that simulates the dynamic delta hedging of a long non-dividend paying European call option. This project investigates the P&L of a hedging strategy and how it is driven by the difference between the implied volatility, used for option pricing and hedging, and the realised volatility of the simulated path.

In this project we will be exploring the applications of a form of delta hedging wherein we maintain a delta neutral portfolio in order to isolate the affect volatility has on PnL. In this way we are trading volatility rather than stock movement. 

Firstly, I must introduce the Black-Scholes-Merton model used in this project and it's underlying assumptions and the meaning of delta. 

Delta(Δ) measures the sensitivity of the option price to the underlying stock price. Mathematically, it's the partial derivative of option value with respect to stock price. For a call option, Δ ranges from 0 to 1; for a put, from -1 to 0. In Black-Scholes, delta also has a probabilistic interpretation: it can be seen as the approximate probability that the option finishes in the money(eg. -0.25 for put and 0.25 for call delta corresponds to a 25% chance of the option finishing in the money). For call options, delta increases with a stock price increases, reflecting higher probability of the call expiring in the money. For put options, delta decreases with an increase in stock price, reflecting the lower probability of the put expiring in the money. This projects focus is on delta hedging a call European option, a type of option which can only be executed at maturity, but the code can be adapted for a European put option instead.

The model used to price the call and calculate deltas is the Black-Scholes-Merton model. This model assumes that stock price follows a geometric Brownian motion, meaning returns are normally distributed and prices evolve continuously with both a deterministic drift term $\mu S_t dt$ and a stochastic term $\sigma S_t dW_t$ driven by Weiner process $W_t$:

$$
dS_t= \mu S_t dt+ \sigma S_t dW_t
$$

By defining function F(S,t)=ln(S), we can apply Ito's Lemma and after integrating over time from 0 to t we come to this equation:

$$
\ln\left(\frac{S_t}{S_0}\right) = \left(\mu - \tfrac{1}{2}\sigma^2\right)t + \sigma W_t
$$

From this, it is clear that due to the derministic dt term and by the fact that $\sigma dW_t$ has a mean of 0 and a variance of $\sigma^2 T$, $\ln\left(\frac{S_t}{S_0}\right)$ follows a normal distribution with mean $(\mu - 0.5 \sigma^2)T$ and a variance of $\sigma^2 T$. We say that stock price has a lognormal distribution because of this. We then exponentiate each side of this equation to get the closed form solution for stock price evolution:

$$
S_t = S_0 \exp\left( \left( \mu - \tfrac{1}{2}\sigma^2 \right)t + \sigma W_t \right)
$$

The derivation above describes the real world evolution of stock price as characterised by the drift term $\mu$. When pricing deivatives such as European call options, we change the measure to the risk-neutral, Q. In this measure, the Black-Scholes assumption of no-arbitrage is satisfied. We will next discuss how path generation is carried out and how call price and delta are calculated in the risk-neutral measure.

To find a continuous time evolution of stock price $S_t$ under the risk-neutral measure, we need to go back to the real-world $dS_t$ equation and replace $dW_t$ with a new brownian motion $\widetilde{W}_t$ with relation to $dW_t$ described by:

$$
d\widetilde{W}_t = dW_t^{\mathbb{P}} + \frac{\mu - r}{\sigma}  dt
$$

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

Thus, the second term becomes

$$
-K e^{-rT} N(d_2).
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

Putting both terms together and discounting by $\(e^{-rT}\)$ yields the Black-Scholes call price:

$$
c_0 = e^{-rT}\Big(S_0 e^{rT} N(d_1) - K N(d_2) e^{rT}\Big) = S_0 N(d_1) - K e^{-rT} N(d_2).
$$

If we want to find the price of a non-dividend paying European put, we can use the put-call parity:

$$
p_0 = c_0 + K e^{-rT} - S_0 = S_0 N(d_1) - K e^{-rT} N(d_2) + K e^{-rT} - S_0.
$$

Simplifying, we obtain:

$$
p_0 = K e^{-rT} \big(1 - N(d_2)\big) - S_0 \big(1 - N(d_1)\big) = K e^{-rT} N(-d_2) - S_0 N(-d_1),
$$

since $(1 - N(x) = N(-x)$ for the standard normal distribution.

Now tha


structure: explain how delta is calculated 
explain trading strategy
explain method to analyse PnLs

