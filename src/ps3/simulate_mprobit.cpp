//  Simulate data for estimation of multinomial probit model

#include <iostream>
#include <random>

int main() {
  const int Nobservations = 1000;

  // Define the main parameters of the model
  const std::vector<double> betaj = {0.0, 2.5, 5.0};
  const double betatime = -1.0;
  const double betacost = -1.5;
  const double sigma11 = 1.0;
  const double sigma22 = 1.0;
  const double sigma33 = 1.0;
  const double rho12 = 0.5 / sqrt(sigma11 * sigma22);

  // Open file to write data
  FILE* pFile;
  pFile = fopen("sim_mprobit_data.txt", "w");

  // Set up random number generators
  int seed = 7;
  std::mt19937_64 mt(seed);
  std::normal_distribution<double> stdnormal(0.0, 1.0);

  std::cout << "Simulating data...\n";

  // Simulate 1000 observations and save them to the data file. Use iobs as the
  // observation ID
  for (int iobs = 0; iobs < Nobservations; iobs++) {
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

    // Generate observables and representative utility
    std::vector<double> time(3, 0.0);
    std::vector<double> cost(3, 0.0);
    std::vector<double> Uj(3, 0.0);
    for (int jalt = 0; jalt < 3; jalt++) {
      // Generate Travel Time
      time[jalt] = exp(stdnormal(mt));

      // Generate Trip Cost
      cost[jalt] = exp(stdnormal(mt));

      // Calculate the utility for alternative jalt
      Uj[jalt] = betaj[jalt] + betatime * time[jalt] + betacost * cost[jalt] +
                 unobsU[jalt];
    }

    // Check which alternative has the highest utility
    int decision = 3;
    if ((Uj[0] > Uj[1]) && (Uj[0] > Uj[2]))
      decision = 1;
    else if ((Uj[1] > Uj[0]) && (Uj[1] > Uj[2]))
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
