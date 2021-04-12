//  Simulate data for estimation of a nested logit model

#include <iostream>
#include <random>

int main() {
  const int Nobservations = 1000;

  // Define the main parameters of the model
  const std::vector<double> betaj = {0.0, 2.5, 5.0};
  const double betatime = -1.0;
  const double betacost = -1.5;
  const double lambda12 = 0.5;

  // Open file to write data
  FILE *pFile;
  pFile = fopen("sim_mlogit_data.txt", "w");

  // Set up random number generators
  int seed = 7;
  std::mt19937_64 mt(seed);
  std::normal_distribution<double> stdnormal(0.0, 1.0);
  std::uniform_real_distribution<double> unifdist(0.0, 1.0);

  std::cout << "Simulating data...\n";

  // Simulate 1000 observations and save them to the data file.
  // 'iobs' denotes the observation ID
  for (int iobs = 0; iobs < Nobservations; iobs++) {
    // Generate observables and representative utility
    std::vector<double> time(3, 0.0);
    std::vector<double> cost(3, 0.0);
    std::vector<double> Vj(3, 0.0);
    for (int jalt = 0; jalt < 3; jalt++) {
      // Generate Travel Time
      time[jalt] = exp(stdnormal(mt));

      // Generate Trip Cost
      cost[jalt] = exp(stdnormal(mt));

      // Calculate representative utility for alternative jalt
      Vj[jalt] = betaj[jalt] + betatime * time[jalt] + betacost * cost[jalt];
    }

    // Calculate choice probabilities for nested logit
    std::vector<double> choiceprob(3, 0.0);

    double group1 = exp(Vj[0] / lambda12) + exp(Vj[1] / lambda12);
    double denom = pow(group1, lambda12) + exp(Vj[2]);

    for (int jalt = 0; jalt < 2; jalt++) {
      choiceprob[jalt] =
          exp(Vj[jalt] / lambda12) * pow(group1, lambda12 - 1.0) / denom;
    }
    choiceprob[2] = exp(Vj[2]) / denom;

    printf("Probabilitys are %5.3f %5.3f %5.3f %5.3f\n", choiceprob[0],
           choiceprob[1], choiceprob[2],
           (choiceprob[0] + choiceprob[1] + choiceprob[2]));

    // Draw from uniform distribution to generate choice
    double unidraw = unifdist(mt);

    int decision = 3;
    if (unidraw < choiceprob[0])
      decision = 1;
    else if (unidraw < (choiceprob[0] + choiceprob[1]))
      decision = 2;

    // Save simulated data to file.
    fprintf(pFile, "%5d ", iobs);
    for (int jalt = 0; jalt < 3; jalt++) fprintf(pFile, "%10.8f ", time[jalt]);
    for (int jalt = 0; jalt < 3; jalt++) fprintf(pFile, "%10.8f ", cost[jalt]);
    fprintf(pFile, "%5d ", decision);

    // Start a new line once all the data for this observation has been written
    fprintf(pFile, "\n");
  }

  // Close file
  fclose(pFile);

  std::cout << "Done simulating data.\n";
  return 0;
}
