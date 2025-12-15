# project-work

# Approaches
We used multiple approaches to solve the gold collection problem.

## Merge Approach

The merge approach is a constructive heuristic that builds efficient multi-stop routes by combining single-city trips.

### Algorithm Overview

1. **Single-City Trip Generation**
   - First, compute the optimal round trip for each city individually (depot → city → depot)
   - For the outbound journey (unloaded), use Dijkstra's algorithm to find the shortest path
   - For the return journey (loaded with gold), use A* search with a weighted heuristic that accounts for the gold being carried
   - Each trip's cost is calculated using the weight-dependent cost function: `cost = dist + (dist × α × weight)^β`

2. **Trip Sorting**
   - Sort all single-city trips by path length in descending order
   - Longer trips are processed first because they offer more opportunities for merging

3. **Greedy Merging**
   - For each main trip, examine cities that lie on its return path
   - For each candidate city on the return path:
     - Calculate the **marginal cost**: the additional cost of picking up that city's gold versus continuing with current weight
     - Compare marginal cost with the standalone cost of doing that city separately
     - If `marginal_cost < standalone_cost`, merge the city into the current trip
   - Mark merged cities as "excluded" to prevent them from being used in other trips

4. **Path Optimization ($\beta > 1$ only)**
   - When $\beta > 1$ we can apply another optimization explained in detail below

### Key Features

- **A\* Heuristic**: Uses Euclidean distance as a lower bound for the weighted travel cost
- **Distance Caching**: Pre-computes all edge distances in a path for efficient cost recalculation
- **Greedy Selection**: Makes locally optimal merge decisions without backtracking
- **Non-worse performance**: The approach of the algorithm guarantees for all instances that the path found is at least as good as the baseline

### Complexity

- Single-city trip generation: $O(n \times E \log V)$ where $n$ is the number of cities
- Merging phase: $O(n^2 \times L)$ where $L$ is the average path length
- Overall: Polynomial time, making it suitable for large instances

## Evolutionary Approach

Evolutionary Approach with Smart DecodingThis project implements a solution based on an **Evolutionary Algorithm (EA)** designed to optimize the collection of gold across a graph with weight-dependent travel costs. The core innovation lies in the separation between the **visitation order** (evolved by the GA) and the **logistics of unloading** (handled by a deterministic greedy decoder).

### Algorithm Overview

Genotype RepresentationThe problem is modeled similarly to a Traveling Salesman Problem (TSP):

* **Genotype:** A permutation of city indices P = [c_1, c_2, ..., c_N].
* This represents the *intent* to visit cities in a specific sequence, ignoring (at this stage) the constraints of weight and depot returns.

#### Smart Phenotypic Decoder (The Greedy Strategy)
The genotype is transformed into a feasible path (phenotype) through a constructive heuristic that manages the vehicle's load. The algorithm iterates through the city sequence defined by the genotype and makes a locally optimal decision at each step:

1. **State Tracking:** The agent tracks current location, current accumulated weight, and total cost.
2. **Decision Step:** For each next target city c~next~ in the genome:
* **Option A (Direct):** Travel directly from c~current~ to c~next~ carrying the current load.
* **Option B (Unload):** Return to depot c~current~ to 0 (unload weight), then travel 0 to c~next~ (empty).


3. **Cost Evaluation:**
$$*  Cost~direct~= dist(curr, next) + (dist \cdot \alpha \cdot weight)^\beta $$
$$*  Cost~split~= Cost(curr \to 0 | weight) + Cost(0 \to next | 0) $$


4. **Selection:** If Cost~split~ < Cost~direct~, the agent inserts a depot visit into the path. Otherwise, it proceeds directly.
#### Heuristic Injection (Initialization)
To guarantee performance at least equal to a standard baseline, the initial population is not entirely random.

* We inject a single **"Baseline Individual"** whose genome corresponds to visiting cities ordered by their distance from the depot.
* This ensures the evolutionary search starts with a strong upper bound on fitness, preventing "cold start" issues in complex topologies.

#### Evolutionary Operators
* **Selection:** Tournament Selection (ex. k=3) to maintain selection pressure while preserving diversity.
* **Crossover:** Ordered Crossover (OX1), chosen to preserve the relative order of city visits from parents.
* **Mutation:** Hybrid approach combining:
  * **Inversion (2-opt):** Reverses a subsequence, effective for unraveling crossing paths in geometric spaces.
  * **Swap:** Exchanges two random cities, useful for fine-tuning the sequence.
  * *Probability:* 50% Inversion / 50% Swap.

#### Advanced Improvements
To further enhance convergence and solution quality, we implemented:

1.  **Memetic Algorithm (Local Search):**
    *   After mutation, each individual undergoes a **Hill Climbing** phase.
    *   The algorithm attempts to swap adjacent cities in the genome.
    *   If a swap improves fitness, it is kept (First Improvement strategy).
    *   This allows the algorithm to refine solutions locally, while the GA handles global exploration.

2.  **Adaptive Mutation (Stagnation Control):**
    *   The mutation rate is dynamic.
    *   If the best fitness does not improve for a set number of generations (stagnation), the mutation rate increases (up to 0.8).
    *   This "Hyper-mutation" helps the population escape local optima.
    *   Once a new best solution is found, the rate resets to the base value (0.3).

#### Key Features
* **Decoupled Optimization:** The GA optimizes the *topology* (order of cities), while the Smart Decoder optimizes the *capacity* (when to unload).
* **Distance Caching:** Uses pre-computed All-Pairs Shortest Paths (Dijkstra) stored in a hash map, reducing distance lookups from O(E logV) to O(1) during the evolution loop.
* **Non-Worse Guarantee:** Thanks to Heuristic Injection, the algorithm is mathematically guaranteed to never perform worse than a distance-sorted baseline strategy.
* **Adaptive Behavior:** The decoder automatically adapts to changes in \alpha and \beta. If \beta is high (high penalty for weight), the decoder naturally chooses to return to the depot more frequently without changing the GA logic.

#### Complexity
* **Preprocessing:** O(V $\cdot$ (E + V logV)) for computing the distance cache (executed once).
* **Phenotype Reconstruction:** O(N) where N is the number of cities (linear scan of the genome).
* **Evolutionary Step:** O(P $\cdot$ N) per generation, where P is population size.
* **Overall:** The approach is computationally lightweight, allowing for large populations and many generations even on standard hardware.

## Beta optimization
When $\beta > 1$ we can exploit the weights' non linearities to subdivide each round trip in $N$ trips, each one picking $\frac{w}{N}$ (where $w$ is the total gold picked)

To explain the approach we first consider a simpler case, to get used to the notation, then extend to complex paths.

### Simple case
Let's consider first a single round trip where the gold is picked from only one city.
Let's denote 
- $C_{\text{go}}$ as the cost to get to target city (where gold is picked) 
    - $P_{\text{go}}$ is such path represented as pairs `[from, to]` representing pairs of each step
- $C_{return}$ as the cost to return from target city to 
    - $P_{\text{return}}$ as explained before
    - $C_{\text{ret s}}$ is the static part, depending only on geoemtric distance
    - $C_{\text{ret}}(w)$ is the part depending on weight carried
```math
    C_{\text{go}} = \sum_{(i,j) \in P_{\text{go}}} d_{i,j} \\
    C_{\text{return}} = \sum_{(i,j) \in P_{\text{return}}} d_{i,j} + (d_{i,j} \alpha w)^\beta = 
    \sum_{(i,j) \in P_{\text{return}}} d_{i,j} + 
    \sum_{(i,j) \in P_{\text{return}}} (d_{i,j} \alpha w)^\beta =  C_{\text{ret s}} + C_{\text{ret}}(w)
```

Finally the total cost of the round trip can be computed ad
```math
    C = C_{\text{go}} + C_{\text{ret s}} + C_{\text{ret}}(w)
```

Now we consider the cost picking $\frac{w}{N}$ gold at each round-trip, repeated $N$ times

```math
    C(N) = N \left( C_{\text{go}} + C_{\text{ret s}} + C_{\text{ret}} \left(\frac{w}{N} \right) \right )
```

We want to find $N^*$ such that it minimizes the cost.

To find such value, we treat $N$ as a continous variable and differentiate w.r.t. $N$
```math
    C'(N) = C_{\text{go}} + C_{\text{ret s}} + (1 - \beta)N^{-\beta}(\alpha w)^\beta \sum_{(i, j) \in P_{\text{ret}}} d_{i,j}^\beta = 0 
```

Solving the equation yields

```math
    N^* = \alpha w \left ( \frac{\beta - 1} {C_{\text{ret s}} + C_{\text{go}} } \sum_{(i, j) \in P_{\text{ret}}} d_{i,j}^\beta \right ) ^ {\frac{1}{\beta}}
```


#### Second Order Condition (Proof of Minimality)

To strictly prove that $N^*$ represents a **minimum** cost (and not a maximum or an inflection point), we must examine the **second derivative** of the cost function, $C''(N)$.

We differentiate $C'(N)$ with respect to $N$ once again. Note that the static costs ($C_{\text{go}} + C_{\text{ret}}$) are constants, so their derivative is zero. We therefore focus on the other term.

> *(Where $K = (\alpha w)^\beta \sum d^\beta$ is a constant positive term representing weights and distances.)*

Applying the power rule, we obtain:

```math
C''(N) = (1 - \beta)(-\beta) K N^{-\beta - 1}
```

---

#### Analyzing the Sign of $C''(N)$

We analyze the sign of each component under the constraint that **$\beta > 1$** (which is a precondition for this optimization):

1. **$(1 - \beta)$**: since $\beta > 1$, this term is **negative** (< 0).
2. **$(-\beta)$**: since $\beta > 1$, this term is **negative** (< 0).
3. **$N^{-\beta - 1}$**: since $N$ (number of trips) is positive, this term is **positive** (> 0).
4. **$K$**: distances and weights are positive, so this term is **positive** (> 0).

Multiplying these signs together yields:

```math
C''(N) > 0 \text{ } \quad \forall N > 0 \; \text{ when } \beta > 1
```

---

### Conclusion

Since the second derivative is strictly positive, the cost function $C(N)$ is **strictly convex** (it has a $\cup$-shaped profile).

Consequently, the stationary point $N^*$ found where $C'(N) = 0$ is guaranteed to be the **global minimum** of the cost function.


**Note**
- Given the scale of values of $w$ in the context, it is almost always beneficial to split a path into multiple subpaths
- From the formula, it is clear the constraint of $\beta > 1$, otherwise the problem becomes linear or sublinear and the exploit is never beneficial

### General case
In the general case we consider a single round trip in which gold can be picked from multiple cities in the same trip.

We subdivide the path into subpaths where gold carried is constant.

For example consider the path `[0, 2, 4, 5(G), 3, 6(G), 2, 0]`, we can subdivide it into 3 subpaths carrying fixed gold
```math
    S = \{ \{ 0, 2, 4, 5\}, \{ 5, 3, 6 \}, \{ 6, 2, 0 \} \}
```
Carrying respectively `[0, gold(5), gold(5)+gold(6)]` total gold

Considering the cost from a single path to introduce some notation
```math
    C_s = \sum_{(i,j) \in s} d_{i,j} + (d_{i,j} w_s \alpha)^\beta = C_{\text{s static}} + C_{s \text{dyn}}(w_s)
```

We can express the total cost of a single round trip as 
```math
    C = \sum_{s \in S} C_s = 
    \sum_{s \in S} \sum_{(i,j) \in S} d_{i,j} + \sum_{s \in S} \sum_{(i,j) \in S} (\alpha w_s d_{i,j})^\beta =
    \sum_{s \in S} C_{\text{s static}} + C_{\text{s dyn}}(w_s)
```

As before we now consider splitting the path in $N$ identical trips, picking $\frac{w}{N}$ gold each

```math
    C(N) = N \left( \sum_{s \in S} C_{\text{s static}} + C_{s \text{dyn}}\left(\frac{w_s}{N}\right) \right) 
    = N \sum_s C_{s \text{static}} + \alpha^\beta N^{-\beta+1} \sum_s w_s^\beta \sum_{(i,j) \in s} d_{i,j}^\beta
```

To simplify a bit the notation we introduce

```math
    C_{s \text{beta static}} (\beta) = \sum_{(i,j) \in s} d_{i,j}^\beta
```

Getting

```math
    C(N)
    = N \sum_s C_{s \text{static}} + \alpha^\beta N^{-\beta+1} \sum_s w_s^\beta C_{s \text{beta static}} (\beta)
```

Once again, differentiating w.r.t. $N$ and setting the derivative to $0$ yields

```math
    C'(N) = \sum_s C_{s \text{static}} + (1 - \beta) \alpha^\beta N^{-\beta} \sum_s w_s^\beta C_{s \text{beta static}} (\beta) = 0
```

Solving for optimal $N^*$ gives

```math
    N^* = \left ( 
        \alpha^\beta (\beta - 1) \frac{\sum_s w_s^\beta C_{s \text{beta static}} (\beta)}{\sum_s C_{s \text{static}}}
    \right)^{\beta^{-1}}
```

### Implementation Strategy

The derived formula for $N^*$ can be applied independently to each round trip in the solution:

1. **Decomposition**: The complete solution (visiting all cities) is naturally decomposed into multiple round trips by the merge algorithm
2. **Independent Optimization**: Each round trip is optimized separately using its own $N^*$ calculated from the formula above
3. **Reconstruction**: The optimized round trips are concatenated to form the final solution

This approach is valid because:
- Each round trip is independent (starts and ends at depot)
- The optimization formula depends only on the trip's internal structure (distances, weights, pickup sequence). The result depends on the importance of the static costs
- No interaction exists between different trips that would prevent independent optimization

### Key Advantages

- **Analytical Solution**: Closed-form formula with no hyperparameters or iterative optimization
- **Computational Efficiency**: $O(L)$ per trip, where $L$ is the trip length
- **(Almost) Guaranteed Improvement**: When $\beta > 1$, splitting almost always reduces cost (superlinear cost function)
- **Exact Optimum**: The continuous relaxation provides the theoretically optimal split count (rounded to nearest integer)