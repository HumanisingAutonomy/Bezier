
#include <Bezier/bezier.h>
#include <Bezier/declarations.h>
#include <gtest/gtest.h>

#include <cstddef>
constexpr auto kVeryClose = 1e-5;
constexpr auto kCloseEnough = 1e-1;
using callback = std::function<double(double)>;

TEST(Bezier, project_point_linear) {
  Bezier::PointVector controls{{0, 0}, {0, 1}};
  Bezier::Curve curve{controls};

  EXPECT_EQ(curve.projectPoint({1, 0.5}), 0.5);
  EXPECT_EQ(curve.projectPoint({1, 1.5}), 1);
  EXPECT_EQ(curve.projectPoint({1, 1.5}, false), 1.5);
}

Eigen::Matrix<double, 100, 2> generateData(const callback &fn) {
  using Vec = Eigen::Vector<double, 100>;
  using Mat = Eigen::Matrix<double, 100, 2>;
  Vec xs = Vec::LinSpaced(100, 0, 5);

  Vec ys;
  for (size_t i = 0; i < 100; i++) {
    ys[i] = fn(xs[i]);
  }

  Mat data;
  data << xs, ys;

  return data;
}

void checkRegression(const callback &fn, Bezier::Curve prediction) {
  const Eigen::Vector<double, 100> ts =
      Eigen::Vector<double, 100>::LinSpaced(100, 0, 1);

  std::vector<Eigen::Vector2d> results;

  for (size_t i = 0; i < static_cast<size_t>(ts.rows()); i++) {
    const auto result = prediction.valueAt(ts[i]);
    const auto expected = fn(result.x());
    results.push_back(result);

    // we are estimating the curve so it just has to be close enough
    EXPECT_NEAR(expected, result.y(), kCloseEnough);
  }
  std::vector<double> t_vector(ts.data(), ts.data() + ts.rows());

  // check that the two overloads output the same values
  const auto expectedBatch = prediction.valueAt(t_vector);
  for (size_t i = 0; i < results.size(); i++) {
    EXPECT_NEAR(expectedBatch(i, 0), results[i].x(), kVeryClose);
    EXPECT_NEAR(expectedBatch(i, 1), results[i].y(), kVeryClose);
  }
}

// checks that the output of fn and the bezier curve match

Bezier::Curve checkRegression(const callback &fn, size_t degree) {
  const auto data = generateData(fn);

  auto prediction = Bezier::Curve::fit(data, degree);
  checkRegression(fn, prediction);

  return prediction;
}

void checkSample(const callback &fn, size_t degree, size_t num) {
  const auto data = generateData(fn);
  const auto prediction = Bezier::Curve::fit(data, degree);

  const auto results = prediction.sample(num);

  const auto start = prediction.valueAt(0);
  const auto end = prediction.valueAt(1);

  EXPECT_EQ(results(0, 0), start.x());
  EXPECT_EQ(results(0, 1), start.y());
  EXPECT_EQ(results(results.rows() - 1, 0), end.x());
  EXPECT_EQ(results(results.rows() - 1, 1), end.y());
}

void checkBlend(const callback &fn1, const callback &fn2, size_t degree,
                double alpha) {
  const auto data1 = generateData(fn1);
  const auto data2 = generateData(fn2);
  const auto fnMean = [&fn1, &fn2, &alpha](double x) {
    return (1 - alpha) * fn1(x) + alpha * fn2(x);
  };

  const auto curve1 = Bezier::Curve::fit(data1, degree);
  const auto curve2 = Bezier::Curve::fit(data2, degree);

  const auto curveMean1 = curve1.blend(curve2, alpha);
  const auto curveMean2 = curve2.blend(curve1, alpha);

  checkRegression(fnMean, curveMean1);
  checkRegression(fnMean, curveMean2);
}

TEST(Bezier, test_straight_line_regression) {
  const auto fn = [](double x) { return 2 * x; };

  const auto curve = checkRegression(fn, 3);

  const auto knots = curve.controlPointsMatrix();

  // knots should be predictable in very simple cases i.e. linear functions

  ASSERT_EQ(knots.rows(), 4);
  ASSERT_EQ(knots.cols(), 2);
  EXPECT_NEAR(knots(0, 0), 0, kVeryClose);
  EXPECT_NEAR(knots(0, 1), 0, kVeryClose);
  EXPECT_NEAR(knots(1, 0) * 2, knots(1, 1), kVeryClose);
  EXPECT_NEAR(knots(2, 0) * 2, knots(2, 1), kVeryClose);
  EXPECT_NEAR(knots(3, 0) * 2, knots(3, 1), kVeryClose);
}

TEST(Bezier, test_sin_regression) {
  const auto fn = [](double x) { return std::sin(x); };

  checkRegression(fn, 4);
}

TEST(Bezier, test_mean_linear) {
  const auto fn1 = [](double x) { return 2 * x; };
  const auto fn2 = [](double x) { return 2 * x + 1; };

  for (double b = 0; b < 1; b += 0.1)
    checkBlend(fn1, fn2, 2, 0.5);
}

TEST(Bezier, test_mean_sin_cos) {
  const auto fn1 = [](double x) { return std::sin(x); };
  const auto fn2 = [](double x) { return std::cos(x); };

  for (double b = 0; b < 1; b += 0.1)
    checkBlend(fn1, fn2, 4, 0.5);
}

TEST(Bezier, test_mean_sin_linear) {
  const auto fn1 = [](double x) { return std::sin(x); };
  const auto fn2 = [](double x) { return x * 2; };

  for (double b = 0; b < 1; b += 0.1)
    checkBlend(fn1, fn2, 4, 0.5);
}

TEST(Bezier, test_sample) {
  const auto fn = [](double x) { return std::sin(x); };

  checkSample(fn, 4, 100);
}

TEST(Bezier, test_throws_with_not_enough_data) {
  constexpr auto D = 2;
  auto data = Eigen::Matrix<double, D + 2, 2>::Zero();
  // up to d+1 throws
  for (size_t i = 0; i < D + 1; i++) {
    const auto subData = data.block(0, 0, i, 2);
    EXPECT_THROW(Bezier::Curve::fit(subData, 2), std::invalid_argument);
  }

  // d+2 should work
  Bezier::Curve::fit(data, 2);
}
