
#include <Bezier/bezier.h>
#include <Bezier/declarations.h>
#include <gtest/gtest.h>

TEST(Bezier, project_point_linear) {
  Bezier::PointVector controls{{0, 0}, {0, 1}};
  Bezier::Curve curve{controls};

  EXPECT_EQ(curve.projectPoint({1, 0.5}), 0.5);
  EXPECT_EQ(curve.projectPoint({1, 1.5}), 1);
  EXPECT_EQ(curve.projectPoint({1, 1.5}, false), 1.5);
}
