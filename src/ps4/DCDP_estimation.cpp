/*
Problem Set 4: Discrete Choice Dynamic Programming Models

II) Estimate the model of female labor market participation
via maximum likelihood using the simulated data from part I
*/

#include <assert.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "nlopt.hpp"

// Function that calculates a normal CDF
double normalCDF(double value);

const double PI = 3.14159265358979323846;
const int N_MCDRAWS = 1000;

// Create class that contains data to pass to objective function
class Estdata {
 public:
  // Define vectors of variables in the data
  int numobs;
  std::vector<int> obsid, educ, nchild, working;
  std::vector<double> wage;
};

// The objective function
double objfunc(const std::vector<double> &x, std::vector<double> &grad,
               void *Domain);

int main() {
  const int nobs = 1000;
  const unsigned int nparam = 4;

  // Declare mydata variable that will contain our data
  Estdata mydata;

  // Set the size of the vectors to the size of our dataset
  mydata.numobs = nobs;
  mydata.obsid.resize(nobs, 0);
  mydata.educ.resize(nobs, 0);
  mydata.nchild.resize(nobs, 0);
  mydata.working.resize(nobs, 0);
  mydata.wage.resize(nobs, 0.0);

  printf("Reading in data\n");

  // Create ifstream to read data from file
  std::ifstream infile;
  infile.open("sim_sdcp_data.txt");

  // Loop over observations and read in data
  for (int iobs = 0; iobs < nobs; iobs++) {
    // Read in four numbers
    infile >> mydata.obsid.at(iobs);
    infile >> mydata.educ.at(iobs);
    infile >> mydata.nchild.at(iobs);
    infile >> mydata.working.at(iobs);
    infile >> mydata.wage.at(iobs);

    // Check that we haven't gotten to the end of the file before we expected
    if (!infile.good()) {
      std::cout << "***ABORTING: Problem reading in data\n";
      assert(0);
    }
    // Print out the data to make sure it looks right
    //  printf("%5d %10d %16.10f
    //  %10d\n",mydata.obsid.at(iobs),mydata.educ.at(iobs),mydata.latentvar.at(iobs),mydata.dec.at(iobs));
    if (iobs < 10) {
      printf("%5d ", mydata.obsid.at(iobs));
      printf("%5d %5d %5d %10.8f", mydata.educ.at(iobs), mydata.nchild.at(iobs),
             mydata.working.at(iobs), mydata.wage.at(iobs));
    }
  }

  // Here we set up the optimizer and minimize the likelihood
  nlopt::opt opt(nlopt::LN_COBYLA, nparam);

  // If there are no bounds on parameters, then this is not necessary. use
  // "HUGE_VAL" if you want it to be unconstrained on one side
  std::vector<double> lb(nparam, -HUGE_VAL);
  // lb[0] = 0.0;
  lb[3] = 0.0;
  // lb[4] = -0.95;
  // lb[5] = 0.0;
  opt.set_lower_bounds(lb);

  std::vector<double> ub(nparam, HUGE_VAL);
  // ub[0] = 0.0;
  // ub[4] = 0.95;
  opt.set_upper_bounds(ub);

  // Tell the optimizer where it can find the function that will
  // calculate the objective function and the data object
  opt.set_max_objective(objfunc, &mydata);

  // Here we can change settings of the optimizer
  // opt.set_maxeval(nobs);
  opt.set_xtol_rel(1.0e-10);

  // Create variables containing parameters and value of objective
  std::vector<double> param(nparam, 1.0);
  double Value;

  // Set initial values for parameters
  param.at(0) = -0.01;
  param.at(1) = 0.05;
  param.at(2) = -0.01;
  param.at(3) = 0.5;
  //    param.at(4) = 0.1;
  //    param.at(5) = 0.5;

  // Optimize
  opt.optimize(param, Value);

  printf("*****optimum found:\n");
  printf("beta_educ =     %7.3f\n", param[0]);
  printf("gamma =     %7.3f\n", param[1]);
  printf("beta_n = %7.3f\n", param[2]);
  printf("sigma_eta =  %7.3f\n", param[3]);
  // printf("rho12 =     %7.3f\n",param[4]);
  // printf("sigma_eps = %7.3f\n",param[5]);

  return 0;
}

double objfunc(const std::vector<double> &x, std::vector<double> &grad,
               void *MYdata) {
  double betaeduc = x[0];
  double gamma = x[1];
  double betan = x[2];
  double sigmaeta = x[3];
  //  double rhoepseta = x[4];
  double rhoepseta = 0.0;

  // double sigmaeps = x[5];
  double sigmaeps = 1.0;

  // Need to do some fancy stuff ("cast") to get a pointer to the data
  Estdata *mydata = reinterpret_cast<Estdata *>(MYdata);

  double objective = 0.0;

  // Loop over the observations and add the log-likelihood up
  for (int iobs = 0; iobs < mydata->numobs; iobs++) {
    if (mydata->working[iobs] == 1) {
      // First calculate wage part of density
      double exponentWage = (mydata->wage[iobs] - gamma * mydata->educ[iobs]) *
                            (mydata->wage[iobs] - gamma * mydata->educ[iobs]) /
                            (-2.0 * sigmaeta * sigmaeta);
      double wagedensity = exp(exponentWage) / sigmaeta / sqrt(2.0 * PI);

      // Calculate choice probability
      double udiff = mydata->wage[iobs] - betaeduc * mydata->educ[iobs] -
                     betan * mydata->nchild[iobs];

      // Adjust udiff for correlation between unobservables
      udiff += -1.0 * rhoepseta * sigmaeps *
               (mydata->wage[iobs] - gamma * mydata->educ[iobs]) / sigmaeta;

      double choiceprob =
          normalCDF(udiff / (sigmaeta * sqrt(1.0 - rhoepseta * rhoepseta)));

      // Calculate contribution to likelihood
      double totlikelihood = choiceprob * wagedensity;

      // Make sure we don't take the log of zero
      if (totlikelihood < 1e-10) totlikelihood = 1.0e-10;

      objective += log(totlikelihood);
    } else {
      double udiff = (gamma - betaeduc) * mydata->educ[iobs] -
                     betan * mydata->nchild[iobs];
      double choiceprob = normalCDF(-1.0 * udiff /
                                    sqrt(sigmaeta * sigmaeta -
                                         2.0 * rhoepseta * sigmaeps * sigmaeta +
                                         sigmaeps * sigmaeps));
      // Make sure we don't take the log of zero
      if (choiceprob < 1e-10) choiceprob = 1.0e-10;

      objective += log(choiceprob);
    }
  }

  printf("Evaluating at ");
  for (int iparam = 0; iparam < x.size(); iparam++)
    printf("%16.10f", x[iparam]);
  printf(", objective=%16.10f\n", objective);

  return objective;
}

double normalCDF(double value) { return 0.5 * erfc(-value * M_SQRT1_2); }
