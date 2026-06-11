#include "codegen_bridge.hpp"
#include <polygnition/poly.hpp>

namespace poly = polygnition::polynomial;

extern "C" [[gnu::noinline, gnu::used]] double
polygnition_codegen_static_bridge_target(static_degree20_t p, double x) {
  return poly::evaluate(poly::horner, p, x);
}
