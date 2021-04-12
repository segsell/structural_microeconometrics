/*
Problem Set 2

Estimate probit using NLopt:
http://ab-initio.mit.edu/wiki/index.php/NLopt_Reference#The_nlopt_opt_obje

See here for an overview of available optimizers:
https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/

{G,L}{N,D}_xxxx, where G/L denotes global/local optimization and
N/D denotes derivative-free/gradient-based algorithms, respectively.
*/

#include <assert.h>

#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

#include "nlopt.hpp"

/*
Problem 4:
Use both gradient-free (e.g. polytope) and gradient optimizers (e.g. steepest
descent, Newton, Newton with line search, BFGS, conjugate gradient methods,
etc.) to numerically solve:

min 100(y − x^2)^2 + (1 − x)^2

Use both numerical and exact derivatives for the gradient optimizers.
Which methods do well (poorly)?
How does convergence depend on the initial value in the different
methods?
Note that (x, y) = (1, 1) is the unique solution.
*/

// The objective function
double objfunc(const std::vector<double> &x, std::vector<double> &grad,
               void *Domain);

// Counter
int counter = 0;

int main() {
  // Set the number of parameters/variables used in optimization
  const unsigned int nparam = 2;

  // Create clock object for timing optimization
  std::clock_t start;
  double duration;

  /*
  Here we set up the optimizer and minimize the likelihood.
  - LN_COBYLA is a gradient-free optimizer
  - LD_MMA : Gradient based
  - LD_LBFGS : Gradient based
  - LN_SBPLX : Subplex (a variant of Nelder-Mead that uses Nelder-Mead on a
   sequence of subspaces)
  - LN_COBYLA GN_DIRECT_L: DIviding RECTangles algorithm
  for global optimization
  */

  // nlopt::opt opt(nlopt::LN_SBPLX, nparam);
  nlopt::opt opt(nlopt::LD_LBFGS, nparam);

  // Give it some guidance about the size of the initial step
  std::vector<double> initialstep(2, 1.0);
  opt.set_initial_step(initialstep);

  // Tell the optimizer where it can find the function that will
  // calculate the objective function and the data object
  opt.set_min_objective(objfunc, NULL);

  // Here we can change settings of the optimizer
  // opt.set_maxeval(nobs);
  opt.set_xtol_rel(1.0e-10);

  // Create variables containing parameters and value of objective
  std::vector<double> param(nparam, 5.0);
  double Value;

  // Set initial values for parameters
  // param.at(0) = -1.0;
  // param.at(1) = 0.1;

  // Optimize
  start = std::clock();
  opt.optimize(param, Value);
  duration = (std::clock() - start) / static_cast<double>(CLOCKS_PER_SEC);

  printf("optimum found: %16.10f %16.10f where f=%16.10e\n", param[0], param[1],
         Value);
  printf("Optimization took %10d evaluations in %5.3e seconds.\n", counter,
         duration);

  return 0;
}

double objfunc(const std::vector<double> &x, std::vector<double> &grad,
               void *MYdata) {
  counter++;

  // Declare objective that will be returned to optimizer
  double objective = 0.0;

  // Calculate objective function
  objective = 100.0 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]) +
              (1 - x[0]) * (1 - x[0]);

  // Calculate gradient
  if (!grad.empty()) {
    grad[0] = -400.0 * (x[0] * x[1] - x[0] * x[0] * x[0]) - 2.0 + 2.0 * x[0];
    grad[1] = 200.0 * (x[1] - x[0] * x[0]);
  }

  // Print out evaluation:
  printf("Evaluating at %16.10f %16.10f, objective=%16.10e\n", x[0], x[1],
         objective);

  // Return objective value to optimizer
  return objective;
}
