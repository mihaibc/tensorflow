/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/lite/kernels/test_util.h"

#include <stdint.h>

#include <initializer_list>
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/array.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/test_delegate_providers.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

TEST(TestUtilTest, ArrayFloatNearFp32) {
  std::vector<float> expected = {0.1, 100.0, 0.0, -1};
  // 9 * 10^-6 abs error should be tolerated by 1e-5 abs error.
  std::vector<float> near = {0.100009, 99.999991, 0.000009, -1.000009};
  // 2 * 10^-5 abs error should not be tolerated by 1e-5 abs error.
  std::vector<float> not_near = {0.10002, 99.99998, 0.00002, -1.00002};

  EXPECT_THAT(near, ElementsAreArray(ArrayFloatNear(expected)));
  auto not_near_matchers = ArrayFloatNear(expected);
  for (auto& matcher : not_near_matchers) {
    matcher = Not(matcher);
  }
  EXPECT_THAT(not_near, ElementsAreArray(not_near_matchers));
}

TEST(TestUtilTest, ArrayFloatNearFp16) {
  std::vector<float> expected = {0.1, 100.0, 0.0, -1};
  // 0.003 abs error or <1% rel error should be tolerated by ArrayFloatNear.
  std::vector<float> near = {0.103, 99.1, 0.003, -1.009};
  // 0.004 abs error or >1% rel error should not be tolerated by ArrayFloatNear.
  std::vector<float> not_near = {0.104, 98.9, 0.004, -1.011};

  // Setup FP16 mode.
  tflite::KernelTestDelegateProviders::Get()->MutableParams()->Set<bool>(
      tflite::KernelTestDelegateProviders::kAllowFp16PrecisionForFp32, true);

  EXPECT_THAT(near, ElementsAreArray(ArrayFloatNear(expected)));
  auto not_near_matchers = ArrayFloatNear(expected);
  for (auto& matcher : not_near_matchers) {
    matcher = Not(matcher);
  }
  EXPECT_THAT(not_near, ElementsAreArray(not_near_matchers));

  // Revoke FP16 mode.
  tflite::KernelTestDelegateProviders::Get()->MutableParams()->Set<bool>(
      tflite::KernelTestDelegateProviders::kAllowFp16PrecisionForFp32, false);
}

TEST(TestUtilTest, FloatingPointEqFp32) {
  // Minimum number that FP32 could represent.
  constexpr float fp32_min = 0x1p-149;

  EXPECT_THAT(std::tuple(0.1, 0.1), FloatingPointEq());
  EXPECT_THAT(std::tuple(100, 100), FloatingPointEq());
  EXPECT_THAT(std::tuple(-1, -1), FloatingPointEq());
  EXPECT_THAT(std::tuple(0, 0), FloatingPointEq());

  EXPECT_THAT(std::tuple(0.1, 0.10000002), Not(FloatingPointEq()));
  EXPECT_THAT(std::tuple(100, 100.00002), Not(FloatingPointEq()));
  EXPECT_THAT(std::tuple(-1, -1.0000002), Not(FloatingPointEq()));
  EXPECT_THAT(std::tuple(0, 4 * fp32_min), Not(FloatingPointEq()));
  EXPECT_THAT(std::tuple(0, -4 * fp32_min), Not(FloatingPointEq()));

  // 2 * 10^-7 relative error should be tolerated by 4ULPs.
  EXPECT_THAT(std::tuple(0.1, 0.10000002), FloatingPointAlmostEq());
  EXPECT_THAT(std::tuple(100, 100.00002), FloatingPointAlmostEq());
  EXPECT_THAT(std::tuple(-1, -1.0000002), FloatingPointAlmostEq());
  EXPECT_THAT(std::tuple(0, 4 * fp32_min), FloatingPointAlmostEq());
  EXPECT_THAT(std::tuple(0, -4 * fp32_min), FloatingPointAlmostEq());

  // 6 * 10^-7 relative error should not be tolerated by 4ULPs.
  EXPECT_THAT(std::tuple(0.1, 0.10000006), Not(FloatingPointAlmostEq()));
  EXPECT_THAT(std::tuple(100, 100.00006), Not(FloatingPointAlmostEq()));
  EXPECT_THAT(std::tuple(-1, -1.0000006), Not(FloatingPointAlmostEq()));
  EXPECT_THAT(std::tuple(0, 5 * fp32_min), Not(FloatingPointAlmostEq()));
  EXPECT_THAT(std::tuple(0, -5 * fp32_min), Not(FloatingPointAlmostEq()));
}

TEST(TestUtilTest, FloatingPointEqFp16) {
  // Minimum number that FP16 could represent.
  constexpr float fp16_min = 0x1p-24;
  // Setup FP16 mode.
  tflite::KernelTestDelegateProviders::Get()->MutableParams()->Set<bool>(
      tflite::KernelTestDelegateProviders::kAllowFp16PrecisionForFp32, true);

  // 0.2% relative error should be tolerated by 4ULPs in FP16.
  EXPECT_THAT(std::tuple(0.1, 0.1002), FloatingPointEq());
  EXPECT_THAT(std::tuple(100, 100.2), FloatingPointEq());
  EXPECT_THAT(std::tuple(-1, -1.002), FloatingPointEq());
  EXPECT_THAT(std::tuple(0, 4 * fp16_min), FloatingPointEq());
  EXPECT_THAT(std::tuple(0, -4 * fp16_min), FloatingPointEq());

  // FloatingPointEq() should behave exactly like FloatingPointAlmostEq() in
  // FP16 mode.
  EXPECT_THAT(std::tuple(0.1, 0.1002), FloatingPointAlmostEq());
  EXPECT_THAT(std::tuple(100, 100.2), FloatingPointAlmostEq());
  EXPECT_THAT(std::tuple(-1, -1.002), FloatingPointAlmostEq());
  EXPECT_THAT(std::tuple(0, 4 * fp16_min), FloatingPointAlmostEq());
  EXPECT_THAT(std::tuple(0, -4 * fp16_min), FloatingPointAlmostEq());

  // 0.6% relative error should not be tolerated by 4ULPs in FP16.
  EXPECT_THAT(std::tuple(0.1, 0.1006), Not(FloatingPointEq()));
  EXPECT_THAT(std::tuple(100, 100.6), Not(FloatingPointEq()));
  EXPECT_THAT(std::tuple(-1, -1.006), Not(FloatingPointEq()));
  EXPECT_THAT(std::tuple(0, 5 * fp16_min), Not(FloatingPointEq()));
  EXPECT_THAT(std::tuple(0, -5 * fp16_min), Not(FloatingPointEq()));

  // FloatingPointEq() should behave exactly like FloatingPointAlmostEq() in
  // FP16 mode.
  EXPECT_THAT(std::tuple(0.1, 0.1006), Not(FloatingPointAlmostEq()));
  EXPECT_THAT(std::tuple(100, 100.6), Not(FloatingPointAlmostEq()));
  EXPECT_THAT(std::tuple(-1, -1.006), Not(FloatingPointAlmostEq()));
  EXPECT_THAT(std::tuple(0, 5 * fp16_min), Not(FloatingPointAlmostEq()));
  EXPECT_THAT(std::tuple(0, -5 * fp16_min), Not(FloatingPointAlmostEq()));

  // Revoke FP16 mode.
  tflite::KernelTestDelegateProviders::Get()->MutableParams()->Set<bool>(
      tflite::KernelTestDelegateProviders::kAllowFp16PrecisionForFp32, false);
}

TEST(TestUtilTest, QuantizeVector) {
  std::vector<float> data = {-1.0, -0.5, 0.0, 0.5, 1.0, 1000.0};
  auto q_data = Quantize<uint8_t>(data, /*scale=*/1.0, /*zero_point=*/0);
  std::vector<uint8_t> expected = {0, 0, 0, 1, 1, 255};
  EXPECT_THAT(q_data, ElementsAreArray(expected));
}

TEST(TestUtilTest, QuantizeVectorScalingDown) {
  std::vector<float> data = {-1.0, -0.5, 0.0, 0.5, 1.0, 1000.0};
  auto q_data = Quantize<uint8_t>(data, /*scale=*/10.0, /*zero_point=*/0);
  std::vector<uint8_t> expected = {0, 0, 0, 0, 0, 100};
  EXPECT_THAT(q_data, ElementsAreArray(expected));
}

TEST(TestUtilTest, QuantizeVectorScalingUp) {
  std::vector<float> data = {-1.0, -0.5, 0.0, 0.5, 1.0, 1000.0};
  auto q_data = Quantize<uint8_t>(data, /*scale=*/0.1, /*zero_point=*/0);
  std::vector<uint8_t> expected = {0, 0, 0, 5, 10, 255};
  EXPECT_THAT(q_data, ElementsAreArray(expected));
}

TEST(DimsAreMatcherTestTensor, ValidOneD) {
  TensorUniquePtr t = BuildTfLiteTensor(kTfLiteInt32, {2}, kTfLiteDynamic);
  EXPECT_THAT(t.get(), DimsAre({2}));
}

TEST(DimsAreMatcherTestTensor, ValidTwoD) {
  TensorUniquePtr t = BuildTfLiteTensor(kTfLiteInt32, {2, 3}, kTfLiteDynamic);
  EXPECT_THAT(t.get(), DimsAre({2, 3}));
}

TEST(DimsAreMatcherTestTensor, ValidScalar) {
  TensorUniquePtr t =
      BuildTfLiteTensor(kTfLiteInt32, std::vector<int>{}, kTfLiteDynamic);
  EXPECT_THAT(t.get(), DimsAre({}));
}

TEST(DimsAreMatcherTestArray, ValidOneD) {
  IntArrayUniquePtr arr = BuildTfLiteArray({2});
  EXPECT_THAT(arr.get(), DimsAre({2}));
}

TEST(DimsAreMatcherTestArray, ValidTwoD) {
  IntArrayUniquePtr arr = BuildTfLiteArray({2, 3});
  EXPECT_THAT(arr.get(), DimsAre({2, 3}));
}

TEST(DimsAreMatcherTestArray, ValidScalar) {
  IntArrayUniquePtr arr = BuildTfLiteArray({});
  EXPECT_THAT(arr.get(), DimsAre({}));
}

}  // namespace
}  // namespace tflite
