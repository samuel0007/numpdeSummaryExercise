/**
 * @file summaryexercise_test.cc
 * @brief NPDE homework SummaryExercise code
 * @author Oliver Rietmann, Erick Schulz
 * @date 01.01.2020
 * @copyright Developed at ETH Zurich
 */

#include "../summaryexercise.h"

#include <gtest/gtest.h>

#include <Eigen/Core>

namespace SummaryExercise::test {

TEST(SummaryExercise, dummyFunction) {
  double x = 0.0;
  int n = 0;

  Eigen::Vector2d v = SummaryExercise::dummyFunction(x, n);

  Eigen::Vector2d v_ref = {1.0, 1.0};

  double tol = 1.0e-8;
  ASSERT_NEAR(0.0, (v - v_ref).lpNorm<Eigen::Infinity>(), tol);
}

}  // namespace SummaryExercise::test
