/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <routing/routing_test.cuh>

namespace cuopt {
namespace routing {
namespace test {

// static std::vector<file_params> test_vec{{"cvrptw/R2_2_8.TXT", 17442.29, 19}};
// static std::vector<file_params> test_vec{{"solomon/In/c101_25.txt", 17442.29, 19}};
static std::vector<file_params> test_vec{{"cvrptw/R1_10_1.TXT", 29613.27, 100}};
TEST_P(float_regression_test_t, DUMMY) { test_cvrptw(); }
INSTANTIATE_TEST_SUITE_P(simple_test, float_regression_test_t, ::testing::ValuesIn(test_vec));

TEST_P(regression_routing_test_tsp_t, TSP) { test_tsp(); }
TEST_P(regression_routing_test_acvrp_t, ACVRP) { test_acvrp(); }
TEST_P(regression_routing_test_25_t, CVRPTW_25) { test_cvrptw(); }
TEST_P(regression_routing_test_50_t, CVRPTW_50) { test_cvrptw(); }
TEST_P(regression_routing_test_100_t, CVRPTW_100) { test_cvrptw(); }
TEST_P(float_regression_test_t, CVRPTW) { test_cvrptw(); }
TEST_P(regression_routing_test_pickup_t, PICKUP) { test_cvrptw(); }

INSTANTIATE_TEST_SUITE_P(
  l1_tsp,
  regression_routing_test_tsp_t,
  ::testing::ValuesIn(parse_tests(cuopt::test::read_tests("datasets/ref/l1_tsp.txt"))));
INSTANTIATE_TEST_SUITE_P(
  l1_acvrp,
  regression_routing_test_acvrp_t,
  ::testing::ValuesIn(parse_tests(cuopt::test::read_tests("datasets/ref/l1_acvrp.txt"))));
INSTANTIATE_TEST_SUITE_P(
  l1_25,
  regression_routing_test_25_t,
  ::testing::ValuesIn(parse_tests(cuopt::test::read_tests("datasets/ref/l1_25.txt"))));
INSTANTIATE_TEST_SUITE_P(
  l1_50,
  regression_routing_test_50_t,
  ::testing::ValuesIn(parse_tests(cuopt::test::read_tests("datasets/ref/l1_50.txt"))));
INSTANTIATE_TEST_SUITE_P(
  l1_100,
  regression_routing_test_100_t,
  ::testing::ValuesIn(parse_tests(cuopt::test::read_tests("datasets/ref/l1_100.txt"))));
INSTANTIATE_TEST_SUITE_P(
  l1_homberger,
  float_regression_test_t,
  ::testing::ValuesIn(parse_tests(cuopt::test::read_tests("datasets/ref/l1_homberger.txt"))));
INSTANTIATE_TEST_SUITE_P(
  l1_pickup,
  regression_routing_test_pickup_t,
  ::testing::ValuesIn(parse_tests(cuopt::test::read_tests("datasets/ref/l1_pickup.txt"))));
CUOPT_TEST_PROGRAM_MAIN()

}  // namespace test
}  // namespace routing
}  // namespace cuopt
