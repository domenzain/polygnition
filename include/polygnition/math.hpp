#pragma once
#include <concepts>

namespace polygnition::math {
// compile-time friendly factorial
consteval unsigned long long factorial(int n) {
  if (n <= 1)
    return 1;
  unsigned long long res = 1;
  for (int i = 2; i <= n; ++i)
    res *= i;
  return res;
}
} // namespace polygnition::math
