// Problem Set 3: Estimate a Multinomial Probit Model

#include <assert.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "nlopt.hpp"

const int N_MCDRAWS = 10;

// Create class that contains data to pass to objective function
class data {
 public:
  // Define vectors of variables in the data
  int numobs;
  std::vector<int> obsid, dec;
  std::vector<double> triptime;
  std::vector<double> tripcost;
};

// The objective function
double objfunc(const std::vector<double> &x, std::vector<double> &grad,
               void *Domain);

int main() {
  const int nobs = 1000;
  const unsigned int nparam = 5;

  // Declare mydata variable that will contain our data
  data mydata;

  // Set the size of the vectors to the size of our dataset
  mydata.numobs = nobs;
  mydata.obsid.resize(nobs, 0);
  mydata.triptime.resize(3 * nobs, 0);
  mydata.tripcost.resize(3 * nobs, 0);
  mydata.dec.resize(nobs, 0);

  printf("Reading in data\n");

  // Create ifstream to read data from file
  std::ifstream infile;
  infile.open("sim_mprobit_data.txt");

  // Loop over observations and read in data
  for (int iobs = 0; iobs < nobs; iobs++) {
    // Read in four numbers
    infile >> mydata.obsid.at(iobs);
    for (int jalt = 0; jalt < 3; jalt++)
      infile >> mydata.triptime.at(3 * iobs + jalt);
    for (int jalt = 0; jalt < 3; jalt++)
      infile >> mydata.tripcost.at(3 * iobs + jalt);
    infile >> mydata.dec.at(iobs);

    // Check that we haven't gotten to the end of the file before we expected
    if (!infile.good()) {
      std::cout << "***ABORTING: Problem reading in data\n";
      assert(0);
    }
    // Print out the data to make sure it looks right
    // printf("%5d %10d %16.10f
    // %10d\n",mydata.obsid.at(iobs),mydata.educ.at(iobs),mydata.latentvar.at(iobs),mydata.dec.at(iobs));
    if (iobs < 10) {
      printf("%5d ", mydata.obsid.at(iobs));
      for (int jalt = 0; jalt < 3; jalt++)
        printf("%10.8f ", mydata.triptime.at(3 * iobs + jalt));
      for (int jalt = 0; jalt < 3; jalt++)
        printf("%10.8f ", mydata.tripcost.at(3 * iobs + jalt));
      printf("%5d\n", mydata.dec.at(iobs));
    }
  }

  // Set up the optimizer and minimize the likelihood
  // LN_COBYLA is a gradient-free optimizer
  nlopt::opt opt(nlopt::LN_COBYLA, nparam);

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
  param.at(0) = 0.0;
  param.at(1) = 0.0;
  param.at(2) = -2.0;
  param.at(3) = -2.0;
  param.at(4) = 0.5;
  // param.at(5) = 0.0;

  // Optimize
  opt.optimize(param, Value);

  printf("*****optimum found:\n");
  printf("beta2 =     %7.3f\n", param[0]);
  printf("beta3 =     %7.3f\n", param[1]);
  printf("beta_time = %7.3f\n", param[2]);
  printf("beta_cost = %7.3f\n", param[3]);
  printf("sigma_11 =  %7.3f\n", param[4]);
  //    printf("rho12 =     %7.3f\n",param[5]);

  return 0;
}

double objfunc(const std::vector<double> &x, std::vector<double> &grad,
               void *MYdata) {
  std::vector<double> betaj = {0.0, x[0], x[1]};
  double betatime = x[2];
  double betacost = x[3];
  double sigma11 = x[4];

  //    double rho12 = x[5];
  double rho12 = 0.5;
  const double sigma22 = 1.0;
  const double sigma33 = 1.0;

  // Set up Random Number Generator
  int seed = 7;
  std::mt19937_64 mt(seed);
  std::normal_distribution<double> stdnormal(0.0, 1.0);

  // Need to do some fancy stuff ("cast") to get a pointer to the data
  data *mydata = reinterpret_cast<data *>(MYdata);

  double objective = 0.0;

  // Loop over the observations and add up the log-likelihood
  for (int iobs = 0; iobs < mydata->numobs; iobs++) {
    double choiceprob = 0.0;

    for (int imc = 0; imc < N_MCDRAWS; imc++) {
      // Generate unobserved utility for each of the alternatives
      std::vector<double> unobsU(3, -9999.0);

      // First draw from the standard normal
      for (int jalt = 0; jalt < 3; jalt++) unobsU[jalt] = stdnormal(mt);

      // Now transform into a bivariate normal and an independent normal
      // distributions
      unobsU[1] =
          sigma22 * (rho12 * unobsU[0] + sqrt(1.0 - rho12 * rho12) * unobsU[1]);
      unobsU[0] *= sigma11;
      unobsU[2] *= sigma33;

      // Calculate the utility for alternative jalt
      std::vector<double> Uj(3, -9999.0);
      for (int jalt = 0; jalt < 3; jalt++) {
        Uj[jalt] = betaj[jalt] + betatime * mydata->triptime[3 * iobs + jalt] +
                   betacost * mydata->tripcost[3 * iobs + jalt] + unobsU[jalt];
      }

      if ((mydata->dec[iobs] == 1) && (Uj[0] > Uj[1]) && (Uj[0] > Uj[2]))
        choiceprob++;
      if ((mydata->dec[iobs] == 2) && (Uj[1] > Uj[0]) && (Uj[1] > Uj[2]))
        choiceprob++;
      if ((mydata->dec[iobs] == 3) && (Uj[2] > Uj[0]) && (Uj[2] > Uj[1]))
        choiceprob++;
    }

    choiceprob = choiceprob / static_cast<float>(N_MCDRAWS);

    // Make sure we don't take the log of zero
    if (choiceprob < 1e-10) choiceprob = 1.0e-10;
    objective += log(choiceprob);
  }

  printf("Evaluating at ");
  for (int iparam = 0; iparam < x.size(); iparam++)
    printf("%16.10f", x[iparam]);
  printf(", objective=%16.10f\n", objective);

  return objective;
}
