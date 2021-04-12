/*
Problem Set 5: Discrete Choice Dynamic Programming Models (DCDP)
Dynamic Model of Female Labor Supply

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

// Define some constants
const double PI = 3.14159265358979323846;
const int iNkidsMax = 4;
const int iEducMax = 9;
const int iExpMax = 10;
const int tMax = 9;

// Function that calculates a normal CDF
double normalCDF(double value);

// Function that calculates the choice probability
double ProbD(int t, std::vector<double> param, std::vector<int> state,
             std::vector<double> Emax, double &dens);

// Function that calculates the Emax's (includes integral over birth shocks)
void CalcEmax(std::vector<double> param, std::vector<double> probPreg,
              std::vector<double> &Emax);

// Create class that contains data to pass to objective function
class Estdata {
 public:
  // Define vectors of variables in the data
  int numobs, numperiods;
  std::vector<int> obsid, it, educ, nchild, working, exper;
  std::vector<double> wage, probPreg;
};

// The objective function
double objfunc(const std::vector<double> &x, std::vector<double> &grad,
               void *Domain);

int main() {
  const int nobs = 1000;
  const int nperiods = 10;
  const unsigned int nparam = 6;

  // Declare mydata variable that will contain our data
  Estdata mydata;

  // Set the size of the vectors to the size of our dataset
  mydata.numobs = nobs;
  mydata.numperiods = nperiods;
  mydata.obsid.resize(nobs * nperiods, 0);
  mydata.it.resize(nobs * nperiods, 0);
  mydata.educ.resize(nobs * nperiods, 0);
  mydata.nchild.resize(nobs * nperiods, 0);
  mydata.working.resize(nobs * nperiods, 0);
  mydata.exper.resize(nobs * nperiods, 0);
  mydata.wage.resize(nobs * nperiods, 0.0);
  mydata.probPreg.resize(4, 0.0);

  printf("Reading in data\n");

  // Create ifstream to read data from file
  std::ifstream infile;
  infile.open("sim_dcdp_data.txt");

  // Loop over observations and read in data
  // Also calculate the exogenous transition probabilities
  std::vector<int> kidcount_den(4, 0), kidcount_num(4, 0);

  for (int iobs = 0; iobs < nobs; iobs++) {
    for (int iper = 0; iper < nperiods; iper++) {
      // Read in four numbers
      infile >> mydata.obsid.at(iobs * nperiods + iper);
      infile >> mydata.it.at(iobs * nperiods + iper);
      infile >> mydata.educ.at(iobs * nperiods + iper);
      infile >> mydata.nchild.at(iobs * nperiods + iper);
      infile >> mydata.working.at(iobs * nperiods + iper);
      infile >> mydata.wage.at(iobs * nperiods + iper);
      infile >> mydata.exper.at(iobs * nperiods + iper);

      // Count up total periods with each number of kid
      if (iper < nperiods - 1)
        kidcount_den.at(mydata.nchild.at(iobs * nperiods + iper))++;

      // Count up pregancy shocks conditional on number of kids
      if ((iper > 0) && (mydata.nchild.at(iobs * nperiods + iper) ==
                         mydata.nchild.at(iobs * nperiods + iper - 1) + 1))
        kidcount_num.at(mydata.nchild.at(iobs * nperiods + iper - 1))++;

      // Check that we haven't gotten to the end of the file before we expected
      if (!infile.good()) {
        std::cout << "***ABORTING: Problem reading in data\n";
        assert(0);
      }
      // Print out the data to make sure it looks right
      if (iobs < 10) {
        printf("%5d %5d ", mydata.obsid.at(iobs * nperiods + iper),
               mydata.it.at(iobs * nperiods + iper));
        printf("%5d %5d %5d %10.4f %3d\n",
               mydata.educ.at(iobs * nperiods + iper),
               mydata.nchild.at(iobs * nperiods + iper),
               mydata.working.at(iobs * nperiods + iper),
               mydata.wage.at(iobs * nperiods + iper),
               mydata.exper.at(iobs * nperiods + iper));
      }
    }
  }

  printf("Estimates for the children transition probabilities are:\n");
  for (int ikid = 0; ikid < 4; ikid++) {
    mydata.probPreg.at(ikid) = static_cast<float>(kidcount_num.at(ikid)) /
                               static_cast<float>(kidcount_den.at(ikid));
    printf("%6.3f, ", mydata.probPreg.at(ikid));
  }
  printf("\n");

  // Here we set up the optimizer and minimize the likelihood
  nlopt::opt opt(nlopt::LN_COBYLA, nparam);

  // If there are no bounds on parameters, then this is not necessary. use
  // "HUGE_VAL" if you want it to be unconstrained on one side
  std::vector<double> lb(nparam, -HUGE_VAL);
  lb[0] = 0.0;
  lb[1] = 0.0;
  lb[5] = 0.0;
  opt.set_lower_bounds(lb);

  std::vector<double> ub(nparam, HUGE_VAL);
  ub[2] = 0.0;
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
  param.at(0) = 0.05;
  param.at(1) = 0.05;
  param.at(2) = -0.01;
  param.at(3) = 0.05;
  param.at(4) = 0.02;
  param.at(5) = 0.5;

  // "True" values:
  // param.at(0) = 0.1;
  // param.at(1) = 0.1;
  // param.at(2) = -0.001;
  // param.at(3) = 0.75;
  // param.at(4) = 0.05;
  // param.at(5) = 1.0;

  // Optimize
  opt.optimize(param, Value);

  printf("*****optimum found:\n");
  printf("gamma1 =     %7.3f\n", param[0]);
  printf("gamma2 =     %7.3f\n", param[1]);
  printf("gamma3 = %7.3f\n", param[2]);
  printf("pi =  %7.3f\n", param[3]);
  printf("beta_educ =     %7.3f\n", param[4]);
  printf("sigma_eta = %7.3f\n", param[5]);
  // printf("sigma_eps = %7.3f\n",param[6]);
  // printf("delta = %7.3f\n",param[7]);

  return 0;
}

double objfunc(const std::vector<double> &x, std::vector<double> &grad,
               void *MYdata) {
  // Set up Emax and param vector
  std::vector<double> Emax(9 * 9 * 4 * 10, -9999.0), param(8, 0.0);
  for (int ipar = 0; ipar < x.size(); ipar++) param[ipar] = x[ipar];
  param[6] = 1.0;
  param[7] = 0.95;

  double gamma1 = param[0];
  double gamma2 = param[1];
  double gamma3 = param[2];
  double pi = param[3];
  double betaeduc = param[4];
  double sigmaeta = param[5];
  double sigmaeps = param[6];
  double delta = param[7];

  // Need to do some fancy stuff ("cast") to get a pointer to the data
  Estdata *mydata = reinterpret_cast<Estdata *>(MYdata);
  int nperiods = mydata->numperiods;

  // Solve model
  CalcEmax(param, mydata->probPreg, Emax);

  double objective = 0.0;
  std::vector<int> thisState(3, 0);

  // Loop over the observations and add the log-likelihood up
  for (int iobs = 0; iobs < mydata->numobs; iobs++) {
    for (int iper = 0; iper < nperiods; iper++) {
      if ((mydata->obsid.at(iobs * nperiods + iper) != iobs) ||
          (mydata->it.at(iobs * nperiods + iper) != iper)) {
        printf("****observation and period data is not as expected!****\n");
        assert(0);
      }

      int dataindex = iobs * nperiods + iper;
      thisState = {mydata->educ[dataindex], mydata->nchild[dataindex],
                   mydata->exper[dataindex]};

      if (mydata->working[dataindex] == 1) {
        // First calculate wage part of density
        double expectedwage =
            gamma1 * mydata->educ[dataindex] +
            gamma2 * mydata->exper[dataindex] +
            gamma3 * mydata->exper[dataindex] * mydata->exper[dataindex];
        double exponentWage = (mydata->wage[dataindex] - expectedwage) *
                              (mydata->wage[dataindex] - expectedwage) /
                              (-2.0 * sigmaeta * sigmaeta);
        double wagedensity = exp(exponentWage) / sigmaeta / sqrt(2.0 * PI);

        // Calculate choice probability
        double udiff = mydata->wage[dataindex] -
                       pi * mydata->nchild[dataindex] -
                       betaeduc * mydata->educ[dataindex];

        // If it is not the last period, include the emax:
        if (iper < tMax) {
          int emaxStateIndex =
              iper * iEducMax * iNkidsMax * iExpMax +
              (mydata->educ[dataindex] - 10) * iNkidsMax * iExpMax +
              mydata->nchild[dataindex] * iExpMax + mydata->exper[dataindex];
          udiff += delta * (Emax[emaxStateIndex + 1] - Emax[emaxStateIndex]);
        }
        // adjust udiff for correlation between unobservables
        // udiff += -1.0*rhoepseta*sigmaeps*(mydata->wage[dataindex] -
        // gamma*mydata->educ[dataindex])/sigmaeta;

        double choiceprob = normalCDF(udiff / sigmaeps);

        // Calculate contribution to likelihood
        double totlikelihood = choiceprob * wagedensity;

        // Make sure we don't take the log of zero
        if (totlikelihood < 1e-10) totlikelihood = 1.0e-10;

        objective += log(totlikelihood);
      } else {
        // double udiff = (gamma - betaeduc)*mydata->educ[dataindex] -
        // betan*mydata->nchild[dataindex]; double choiceprob =
        // normalCDF(-1.0*udiff/sqrt(sigmaeta*sigmaeta
        // -2.0*rhoepseta*sigmaeps*sigmaeta + sigmaeps*sigmaeps));
        double dens = 0;
        double choiceprob = 1.0 - ProbD(iper, param, thisState, Emax, dens);
        // Make sure we don't take the log of zero
        if (choiceprob < 1e-10) choiceprob = 1.0e-10;
        objective += log(choiceprob);
      }
    }
  }

  printf("Evaluating at ");
  for (int iparam = 0; iparam < x.size(); iparam++)
    printf("%16.10f", x[iparam]);
  printf(", objective=%16.10f\n", objective);

  return objective;
}

double ProbD(int t, std::vector<double> param, std::vector<int> state,
             std::vector<double> Emax, double &dens) {
  double gamma1 = param[0];
  double gamma2 = param[1];
  double gamma3 = param[2];
  double pi = param[3];
  double betaeduc = param[4];
  double sigmaeta = param[5];
  double sigmaeps = param[6];
  double delta = param[7];

  int educ = state[0];
  int nkids = state[1];
  int exper = state[2];

  // Remember that t index for emax goes from 0 to 8 (t=8 is the last period)
  // We don't need emax function in last period
  int emaxStateIndex = t * iEducMax * iNkidsMax * iExpMax +
                       (educ - 10) * iNkidsMax * iExpMax + nkids * iExpMax +
                       exper;

  double udiff = gamma1 * educ + gamma2 * exper + gamma3 * exper * exper -
                 pi * nkids - betaeduc * educ;

  // If it is not the last period, include the emax:
  if (t < tMax) {
    if ((Emax[emaxStateIndex + 1] < -9998.0) ||
        (Emax[emaxStateIndex] < -9998.0)) {
      printf(
          "\n*****Using uninitialized Emax when evaluating time=%3d, "
          "educ=%3d, exper=%3d, nkids=%3d!!*****\n",
          t, educ, nkids, exper);
      assert(0);
    }
    udiff += delta * (Emax[emaxStateIndex + 1] - Emax[emaxStateIndex]);
  }

  // Calculate density and CDF
  double xsiVar = sigmaeta * sigmaeta + sigmaeps * sigmaeps;
  dens = exp(udiff * udiff / (-2.0 * xsiVar)) / sqrt(2.0 * PI * xsiVar);
  return normalCDF(udiff / sqrt(xsiVar));
}

void CalcEmax(std::vector<double> param, std::vector<double> probPreg,
              std::vector<double> &Emax) {
  double gamma1 = param[0];
  double gamma2 = param[1];
  double gamma3 = param[2];
  double pi = param[3];
  double betaeduc = param[4];
  double sigmaeta = param[5];
  double sigmaeps = param[6];
  double delta = param[7];
  double xsiSD = sqrt(sigmaeta * sigmaeta + sigmaeps * sigmaeps);

  // We need to loop over all possible states and time periods
  // backwards in time
  std::vector<int> thisState(3, 0);

  for (int ieduc = 10; ieduc <= 18; ieduc++) {
    for (int itime = tMax; itime > 0; itime--) {
      // This represents the number of kids in period itime-1
      // We will integrate over the pregnancy shocks here
      // We can do this since a pregnancy shock does not enter into current
      // utility
      for (int ikids = 0; (ikids < 4 && ikids < itime); ikids++) {
        for (int iexp = 0; iexp <= itime; iexp++) {
          // printf("%3d %3d %3d %3d ",itime,ieduc,ikids,iexp);

          int emaxStateIndex = (itime - 1) * iEducMax * iNkidsMax * iExpMax +
                               (ieduc - 10) * iNkidsMax * iExpMax +
                               ikids * iExpMax + iexp;

          int nextEmaxStateIndex = (itime)*iEducMax * iNkidsMax * iExpMax +
                                   (ieduc - 10) * iNkidsMax * iExpMax +
                                   ikids * iExpMax + iexp;

          // First consider no pregnancy shock case
          thisState = {ieduc, ikids, iexp};

          double dens = 0.0;
          double thisProbD = ProbD(itime, param, thisState, Emax, dens);

          // printf("ProbD=%5.3f ",thisProbD);

          Emax[emaxStateIndex] = (gamma1 * ieduc + gamma2 * iexp +
                                  gamma3 * iexp * iexp - pi * ikids) *
                                 thisProbD;
          Emax[emaxStateIndex] += (betaeduc * ieduc) * (1.0 - thisProbD);
          Emax[emaxStateIndex] += xsiSD * dens;

          if (itime < 9) {
            if ((Emax[nextEmaxStateIndex + 1] < -9998.0) ||
                (Emax[nextEmaxStateIndex] < -9998.0)) {
              printf(
                  "\n*****Using uninitialized Emax when evaluating "
                  "time=%3d, educ=%3d, exper=%3d, nkids=%3d!!*****\n",
                  itime - 1, ieduc, ikids, iexp);
              assert(0);
            }
            Emax[emaxStateIndex] +=
                delta * Emax[nextEmaxStateIndex + 1] * thisProbD;
            Emax[emaxStateIndex] +=
                delta * Emax[nextEmaxStateIndex] * (1.0 - thisProbD);
          }

          if (ikids < 3) {
            Emax[emaxStateIndex] *= (1.0 - probPreg[ikids]);

            // Increment indices by one kid:
            thisState = {ieduc, ikids + 1, iexp};

            // Increment Emax index by IExpMax to add a kid
            nextEmaxStateIndex += iExpMax;

            dens = 0.0;
            thisProbD = ProbD(itime, param, thisState, Emax, dens);

            Emax[emaxStateIndex] += probPreg[ikids] *
                                    (gamma1 * ieduc + gamma2 * iexp +
                                     gamma3 * iexp * iexp - pi * (ikids + 1)) *
                                    thisProbD;
            Emax[emaxStateIndex] +=
                probPreg[ikids] * (betaeduc * ieduc) * (1.0 - thisProbD);
            Emax[emaxStateIndex] += probPreg[ikids] * xsiSD * dens;

            if (itime < 9) {
              if ((Emax[nextEmaxStateIndex + 1] < -9998.0) ||
                  (Emax[nextEmaxStateIndex] < -9998.0)) {
                printf(
                    "\n*****Using uninitialized Emax when evaluating "
                    "time=%3d, educ=%3d, exper=%3d, nkids=%3d!!*****\n",
                    itime, ieduc, ikids + 1, iexp);
                assert(0);
              }

              Emax[emaxStateIndex] += probPreg[ikids] * delta *
                                      Emax[nextEmaxStateIndex + 1] * thisProbD;
              Emax[emaxStateIndex] += probPreg[ikids] * delta *
                                      Emax[nextEmaxStateIndex] *
                                      (1.0 - thisProbD);
            }
          }
          // printf("Emax=%6.3f\n",Emax[emaxStateIndex]);
        }
      }
    }
  }
}

double normalCDF(double value) { return 0.5 * erfc(-value * M_SQRT1_2); }
