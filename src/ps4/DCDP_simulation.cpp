/*
Problem Set 4: Discrete Choice Dynamic Programming Models

I) Simulate the model for female labor market participation from class
*/

#include <iostream>
#include <random>

int main() {
  const int Nobservations = 1000;

  // Define the main parameters of the model
  const double betaeduc = -0.05;
  // const double betaeduc = -0.00;
  const double betan = 0.5;
  const double gamma = 0.1;
  const double pi = 0.25;
  const double sigmaeps = 1.0;
  const double sigmaeta = 1.0;
  const double rhoepseta = 0.0;

  // Open file to write data
  FILE *pFile;
  pFile = fopen("sim_sdcp_data.txt", "w");

  // Set up random number generators
  int seed = 7;
  std::mt19937_64 mt(seed);
  std::normal_distribution<double> stdnormal(0.0, 1.0);
  std::uniform_int_distribution<int> educdist(10, 18);
  std::uniform_int_distribution<int> nchilddist(0, 3);

  std::cout << "Simulating data...\n";

  // Simulate 1000 observations and save them to the data file.
  // Use iobs as the observation ID
  for (int iobs = 0; iobs < Nobservations; iobs++) {
    // First draw unobserved utility and wage shocks
    double eta = sigmaeta * stdnormal(mt);
    // double eps = sigmaeps*stdnormal(mt);
    double eps = sigmaeps * (rhoepseta * eta / sigmaeta +
                             sqrt(1.0 - rhoepseta * rhoepseta) * stdnormal(mt));

    // Generate observables
    int educ = educdist(mt);
    int nchild = nchilddist(mt);
    double y = stdnormal(mt);

    // Calculate utility of working (d=1) and not working (d=0)
    double U1 = y + gamma * educ + eta - pi * nchild;
    double U0 = y + betaeduc * educ + betan * nchild + eps;

    int di = (U1 > U0);

    double wage = -9999.0;
    if (di == 1) wage = gamma * educ + eta;

    // Save simulated data to file.
    fprintf(pFile, "%5d ", iobs);
    fprintf(pFile, "%5d %5d %5d %10.8f", educ, nchild, di, wage);

    // Make sure to start a new line once all the data for this observation has
    // been written
    fprintf(pFile, "\n");
  }

  // Close file
  fclose(pFile);

  std::cout << "Done simulating data.\n";
  return 0;
}
