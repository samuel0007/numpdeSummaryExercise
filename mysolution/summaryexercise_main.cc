/**
 * @file summaryexercise_main.cc
 * @brief NPDE homework SummaryExercise code
 * @author Oliver Rietmann, Erick Schulz
 * @date 01.01.2020
 * @copyright Developed at ETH Zurich
 */

#include <Eigen/Core>
#include <iostream>

#include "summaryexercise.h"

int main() {
  Eigen::VectorXd v = SummaryExercise::dummyFunction(0.0, 0);

  const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision,
                                         Eigen::DontAlignCols, ", ", "\n");
  std::cout << v.transpose().format(CSVFormat) << std::endl;

  return 0;
}
