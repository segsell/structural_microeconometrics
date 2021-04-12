/*
Problem Set 5: Discrete Choice Dynamic Programming Models (DCDP)
Dynamic Model of Female Labor Supply

I) Simulate a variaton of the model of female labor market participation
discussed in class
*/

#include <assert.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

// Define some constants
const double PI = 3.14159265358979323846;

const int iNkidsMax = 4;
const int iEducMax = 9;
const int iExpMax = 10;
const int tMax = 9;

// Probabilities of a pregnancy shock given n kids
const std::vector<double> probPreg = {0.2, 0.3, 0.2, 0.0};

// Function that calculates a normal CDF
double normalCDF(double value);

// Function that calculates the choice probability
double ProbD(int t, std::vector<double> param, std::vector<int> state,
             std::vector<double> Emax, double &dens);

// Function that calculates the Emax's (includes integral over birth shocks)
void CalcEmax(std::vector<double> param, std::vector<double> &Emax);

int main() {
  const int Nobservations = 1000;

  // Define the main parameters of the model
  const double gamma1 = 0.1;
  const double gamma2 = 0.1;
  const double gamma3 = -0.001;
  const double pi = 0.75;  // 0.75
  const double betaeduc = 0.05;
  const double sigmaeps = 1.0;
  const double sigmaeta = 1.0;
  const double delta = 0.95;

  std::vector<double> param(8, 0.0);

  param[0] = gamma1;
  param[1] = gamma2;
  param[2] = gamma3;
  param[3] = pi;
  param[4] = betaeduc;
  param[5] = sigmaeta;
  param[6] = sigmaeps;
  param[7] = delta;

  // setup four-dimension matrix
  // not all elements will be used
  std::vector<double> Emax(9 * 9 * 4 * 10, -9999.0);

  // Solve model
  CalcEmax(param, Emax);

  // Open file to write data
  FILE *pFile;
  pFile = fopen("sim_dcdp_data.txt", "w");

  // Set up random number generators
  int seed = 7;
  std::mt19937_64 mt(seed);
  std::normal_distribution<double> stdnormal(0.0, 1.0);
  std::uniform_int_distribution<int> educdist(10, 18);
  std::uniform_real_distribution<double> unifdist(0, 1);

  std::cout << "Simulating data...\n";

  // Simulate 1000 observations and save them to the data file.
  // Use iobs as the observation ID
  for (int iobs = 0; iobs < Nobservations; iobs++) {
    // Generate observables
    int educ = educdist(mt);
    int nkids = 0;
    int exper = 0;

    for (int it = 0; it < 10; it++) {
      // First draw unobserved utility and wage shocks
      double eta = sigmaeta * stdnormal(mt);
      double eps = sigmaeps * stdnormal(mt);

      // Calculate wage
      double wage =
          gamma1 * educ + gamma2 * exper + gamma3 * exper * exper + eta;

      // Calculate utility of working (d=1) and not working (d=0)
      double U1 = wage - pi * nkids;
      double U0 = betaeduc * educ + eps;

      if (it < 9) {
        int emaxStateIndex = it * iEducMax * iNkidsMax * iExpMax +
                             (educ - 10) * iNkidsMax * iExpMax +
                             nkids * iExpMax + exper;

        U1 += delta * Emax[emaxStateIndex + 1];
        U0 += delta * Emax[emaxStateIndex];
      }

      int di = (U1 > U0);

      if (di == 0) wage = -9999.0;

      // Save simulated data to file.
      fprintf(pFile, "%5d %5d ", iobs, it);
      // fprintf(pFile,"%5d %5d %5d %10.3f %5d %10.3f %6.3f ", educ,
      // nkids, di, wage, exper,U1,U0);
      fprintf(pFile, "%5d %5d %5d %10.3f %5d", educ, nkids, di, wage, exper);

      if (it < 9) {
        int emaxStateIndex = it * iEducMax * iNkidsMax * iExpMax +
                             (educ - 10) * iNkidsMax * iExpMax +
                             nkids * iExpMax + exper;
        // fprintf (pFile, "%6.3f %6.3f
        // ",Emax[emaxStateIndex+1],Emax[emaxStateIndex]);
      }

      // Make sure to start a new line once all the data for this observation
      // and period has been written
      fprintf(pFile, "\n");

      // Pregnancy shock
      if ((nkids < 3) && (unifdist(mt) < probPreg[nkids])) nkids++;
      if (di == 1) exper++;
    }
  }

  // Close file
  fclose(pFile);

  std::cout << "Done simulating data.\n";
  return 0;
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

void CalcEmax(std::vector<double> param, std::vector<double> &Emax) {
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
          printf("%3d %3d %3d %3d ", itime, ieduc, ikids, iexp);

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

          printf("ProbD=%5.3f ", thisProbD);

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
          printf("Emax=%6.3f\n", Emax[emaxStateIndex]);
        }
      }
    }
  }
}

double normalCDF(double value) { return 0.5 * erfc(-value * M_SQRT1_2); }
