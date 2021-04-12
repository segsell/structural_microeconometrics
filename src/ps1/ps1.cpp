#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <random>

const double pi = 3.14159265358979323846;
const int64_t ninterOper = pow(10, 9);
const int64_t ninterPI = pow(10, 9);

int main(int argc, const char *argv[]) {
  std::cout << "Hello, World!\n";

  // ********************
  // CALCULATE MACHINE PRECISION
  // ********************
  printf("************ CALCULATING MACHINE PRECISION*************\n");

  long double prev_epsilon, EPS = 0.5;
  while ((1.0 + EPS) != 1.0) {
    prev_epsilon = EPS;
    EPS /= 2.0;
  }
  printf("Machine Precision is %10.8Le.\n", prev_epsilon);

  // ********************
  // CALCULATE TIME OF CALCULATIONS
  // ********************
  printf(
      "\n\n************ CALCULATING RELATIVE SPEED OF "
      "OPERATIONS*************\n");

  // Declare variables for timing operations
  std::clock_t start;
  double duration, duration_add;

  // Declare variables for calculating operations
  double operand = 1.0;

  // ADDITION
  start = std::clock();
  for (int i = 0; i < ninterOper; i++) {
    operand += 0.1;
  }
  duration_add = (std::clock() - start) / static_cast<double>(CLOCKS_PER_SEC);
  printf("Calculating %ld additions takes      %5.3e seconds.\n", ninterOper,
         duration_add);

  // MULTIPLICATION
  start = std::clock();
  for (int i = 0; i < ninterOper; i++) {
    operand *= 0.1;
  }
  duration = (std::clock() - start) / static_cast<double>(CLOCKS_PER_SEC);
  printf(
      "Calculating %ld multiplication takes %5.3e seconds or %5.3f longer.\n",
      ninterOper, duration, duration / duration_add);

  // EXPONENTIATION
  start = std::clock();
  for (int i = 0; i < ninterOper; i++) {
    operand *= pow(2.0, 2.0);
  }
  duration = (std::clock() - start) / static_cast<double>(CLOCKS_PER_SEC);
  printf(
      "Calculating %ld exponentiation takes %5.3e seconds or %5.3f longer.\n",
      ninterOper, duration, duration / duration_add);

  // SINE
  start = std::clock();
  for (int i = 0; i < ninterOper; i++) {
    operand *= sin(1.0);
  }
  duration = (std::clock() - start) / static_cast<double>(CLOCKS_PER_SEC);
  printf(
      "Calculating %ld sine takes           %5.3e seconds or %5.3f longer.\n",
      ninterOper, duration, duration / duration_add);

  // ********************
  // CALCULATE PI
  // ********************
  printf("\n\n************ CALCULATING PI*************\n");

  // Declare random number generator
  std::mt19937 mt(17);
  // Declare type of distribution being used
  std::uniform_real_distribution<double> dist(0, 1.0);

  // variable to store number of successes
  int nsuccess = 0;
  start = std::clock();
  for (int i = 0; i < ninterPI; i++) {
    double x = dist(mt);
    double y = dist(mt);
    if (x * x + y * y <= 1.0) nsuccess++;
  }
  duration = (std::clock() - start) / static_cast<double>(CLOCKS_PER_SEC);
  printf("Fraction is equal to %18.16f. Pi=%10.8f\n",
         static_cast<double>(nsuccess) / static_cast<double>(ninterPI),
         4.0 * static_cast<double>(nsuccess) / static_cast<double>(ninterPI));
  printf(
      "MC Pi - true Pi =%10.8f took %5.3e seconds.\n",
      pi - 4.0 * static_cast<double>(nsuccess) / static_cast<double>(ninterPI),
      duration);

  return 0;
}
