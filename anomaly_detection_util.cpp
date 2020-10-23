/*
 * animaly_detection_util.cpp
 *
 * Author: 313566309 Benny Eyov
 */

#include "anomaly_detection_util.h"
#include <math.h>

float avg(float *x, int size, bool flag = false) {
  float sum = 0;
  for (int i = 0; i < size; i++) {
    if (flag) {
      sum += pow(x[i], 2);
    } else {
      sum += x[i];
    }
  }
  return sum / size;
}

// returns the variance of X and Y
float var(float *x, int size) {
  return avg(x, size, true) - pow(avg(x, size, false), 2);
}

// returns the covariance of X and Y
float cov(float *x, float *y, int size) {
  float multArray[size];
  for (int i = 0; i < size; i++) {
    multArray[i] = x[i] * y[i];
  }
  return avg(multArray, size, false) -
         (avg(x, size, false) * avg(y, size, false));
}

// returns the Pearson correlation coefficient of X and Y
float pearson(float *x, float *y, int size) {
  return cov(x, y, size) / (sqrt(var(x, size)) * sqrt(var(y, size)));
}

// performs a linear regression and returns the line equation
Line linear_reg(Point **points, int size) {
  float x[size];
  float y[size];
  for (int i = 0; i < size; i++) {
    x[i] = points[i]->x;
    y[i] = points[i]->y;
  }
  float a = cov(x, y, size) / var(x, size);
  float b = avg(y, size, false) - a * avg(x, size, false);
  return Line(a, b);
}

// returns the deviation between point p and the line equation of the points
float dev(Point p, Point **points, int size) {
  return dev(p, linear_reg(points, size));
}

// returns the deviation between point p and the line
float dev(Point p, Line l) { return abs(p.y - l.f(p.x)); }
