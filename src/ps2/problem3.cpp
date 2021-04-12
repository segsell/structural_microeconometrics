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
#include <random>
#include <vector>

#include "nlopt.hpp"

/*
Problem 3 :
A government stabilizes the supply of a commodity at S = 2,
but allows the price to be determined by the market.
Domestic and export demand for the commodity are given by:

D = theta_1 * P^-1.0
X = theta_2 * P^-0.5

where log(theta_1) and log(theta_2) are normally distributed with
means 0, variances 0.02 and 0.01, respectively, and covariance 0.01.
*/

// Constants for calculations

// Quadrature weights
const std::vector<double> quadnodes{
    -3.436159118837737603327,  -2.532731674232789796409,
    -1.756683649299881773451,  -1.036610829789513654178,
    -0.3429013272237046087892, 0.342901327223704608789,
    1.036610829789513654178,   1.756683649299881773451,
    2.532731674232789796409,   3.436159118837737603327};
const std::vector<double> quadweights = {
    7.64043285523262062916E-6,  0.001343645746781232692202,
    0.0338743944554810631362,   0.2401386110823146864165,
    0.6108626337353257987836,   0.6108626337353257987836,
    0.2401386110823146864165,   0.03387439445548106313617,
    0.001343645746781232692202, 7.64043285523262062916E-6};

// PI
const double pi = 3.14159265358979323846;

// Parameters for calculations:
const int ndraws = 1000;

const double sigmadom = 0.02;
const double sigmaexp = 0.01;
const double sigmarho = 0.01 / sqrt(sigmadom * sigmaexp);

// Objective function
double objfunc(const std::vector<double> &x, std::vector<double> &grad,
               void *Domain);

// Store the thetas used to calculate the market price
std::vector<double> theta(2, 1.0);

// Counter of the number of evaluations
int counter = 0;

int main() {
  const unsigned int nparam = 1;

  std::clock_t start;
  double duration;
  /*
  Set up the optimizer and minimize the likelihood.
  - LN_COBYLA is a gradient-free optimizer.
  - LN_SBPLX : Subplex (a variant of Nelder-Mead that uses Nelder-Mead on a
  sequence of subspaces)
  - LN_COBYLA GN_DIRECT_L: DIviding RECTangles algorithm
  for global optimization
  */
  // nlopt::opt opt(nlopt::LN_SBPLX, nparam);
  nlopt::opt opt(nlopt::LN_COBYLA, nparam);

  /*
  If there are no bounds on parameters, then this is not necessary.
  Use "HUGE_VAL" if we want it to be unconstrained on one side.
  Note that we put a lower bound of zero ere, so that the optimizer does not try
  negative prices.
  */
  // upper bounds
  std::vector<double> lb(nparam, -HUGE_VAL);
  lb[0] = 0.0;
  // lb[1] = -10.0;
  opt.set_lower_bounds(lb);

  // lower bounds
  std::vector<double> ub(nparam, HUGE_VAL);
  // ub[0] = 2.0;
  // ub[1] = 5.0;
  opt.set_upper_bounds(ub);

  // Tell the optimizer where it can find the function that will
  // calculate the objective function and the data object
  opt.set_min_objective(objfunc, NULL);

  // Here we can change settings of the optimizer
  // opt.set_maxeval(nobs);
  opt.set_xtol_rel(1.0e-8);

  // Create variables containing the parameters and value of objective function
  std::vector<double> param(nparam, 1.0);
  double Value;

  // Set initial values for parameters
  // param.at(0) = -1.0;
  // param.at(1) = 0.1;

  //**************************
  // Evaluate the integral using Gauss-Hermite quadrature:
  //**************************
  printf(
      "*****Calculating E[p] and Var[p] using Gauss-Hermite quadrature "
      "integration\n");

  // Calculate the sum of price and price^2 for each integration point
  double expP = 0.0;
  double sumPsq = 0.0;
  start = std::clock();
  // Integrate over the distribution of domestic (idom) and export (iexp) demand
  // shocks
  for (int idom = 0; idom < 10; idom++) {
    for (int iexp = 0; iexp < 10; iexp++) {
      // Calculate the theta for each quadrature node
      // Quadrature nodes integrate over e^(-x^2)
      // So we must do a change of variables in order to integrate over the
      // log-normal distribution where the components are correlated
      theta[0] = exp(sqrt(2.0) * sqrt(sigmadom) * quadnodes[idom]);
      theta[1] = exp(sqrt(2.0) * sqrt(sigmaexp) *
                     (sigmarho * quadnodes[idom] +
                      sqrt(1 - sigmarho * sigmarho) * quadnodes[iexp]));

      // Set initial value of price for optimization:
      param[0] = 1.0;
      // Calculate the price for those values of theta
      // Note that the theta variables are global variables so they will be
      // accessible from objfunc()
      opt.optimize(param, Value);
      // printf("optimum found for node %d, %d: %16.10f where f=%16.10e\n",
      // idom, iexp, param[0], Value);

      // Sum the price and price squared including the quadrature weights for
      // both dimensions
      expP += quadweights[idom] * quadweights[iexp] * param[0];
      sumPsq += quadweights[idom] * quadweights[iexp] * param[0] * param[0];
    }
  }

  // Calculate expected price and variance
  expP = expP / pi;
  double varP = sumPsq / pi - expP;
  printf("E[P] = %f, Var(P)=%f\n", expP, varP);
  duration = (std::clock() - start) / static_cast<double>(CLOCKS_PER_SEC);
  printf("Expectation took %10d evaluations in %5.3e seconds.\n", counter,
         duration);

  //**************************
  // Evaluate the integral using Monte Carlo Integration
  //**************************
  printf("*****Calculating E[p] and Var[p] using Monte Carlo integration\n");

  // Declare random number generator
  std::mt19937 mt(17);

  // Declare type of distribution being used
  std::normal_distribution<double> normdist(0, 1.0);

  // Reset the variables that store the sums
  expP = 0.0;
  sumPsq = 0.0;

  start = std::clock();

  // Evaluate problem using MC Integration:
  for (int idraw = 0; idraw < ndraws; idraw++) {
    // Draw x and y from independent normal distributions
    double x = normdist(mt);
    double y = normdist(mt);

    // Calculate the theta for each monte carlo draw
    // We must do a change of variables in order to integrate over the
    // log-normal distribution where the components are correlated This is
    // almost identical calculation, except there is no sqrt(2.0) here since we
    // are drawing from standard normal, while quadrature integrates over
    // e^(-x^2)
    theta[0] = exp(sqrt(sigmadom) * x);
    theta[1] = exp(sqrt(sigmaexp) *
                   (sigmarho * x + sqrt(1 - sigmarho * sigmarho) * y));

    // Set initial value of parameter:
    param[0] = 1.0;
    opt.optimize(param, Value);
    // printf("optimum found for draw %d: %16.10f where f=%16.10e\n",
    // idraw, param[0], Value);

    // Sum the price and price squared (no weights required for MC integration)
    expP += param[0];
    sumPsq += param[0] * param[0];
  }

  // Calculate expected price and variance
  expP = expP / static_cast<double>(ndraws);
  varP = sumPsq / static_cast<double>(ndraws) - expP;
  printf("E[P] = %f, Var(P)=%f\n", expP, varP);

  duration = (std::clock() - start) / static_cast<double> CLOCKS_PER_SEC;
  printf("Expectation took %10d evaluations in %5.3e seconds.\n", counter,
         duration);

  return 0;
}

double objfunc(const std::vector<double> &x, std::vector<double> &grad,
               void *MYdata) {
  counter++;

  // Declare objective that will be returned to the optimizer
  double objective = 0.0;

  // Calculate objective function
  // We are calculating the squared difference between demand and supply
  // The minimum of this function should solve for the price that equates them
  objective = (2.0 - theta[0] / x[0] - theta[1] / sqrt(x[0]));
  objective = objective * objective;

  // Print out evaluation:
  // printf("Evaluating at %16.10f , objective=%16.10e\n", x[0],objective);

  // Return objective value to optimizer
  return objective;
}
