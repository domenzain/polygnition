#pragma once
#include <cstddef>
#include <polygnition/poly.hpp>
#include <utility>

namespace polygnition_codegen {
namespace poly = polygnition::polynomial;

template <int N, std::size_t... I>
[[nodiscard]] consteval auto make_literal_impl(std::index_sequence<I...>) {
  return poly::literal<static_cast<double>(I + 1)...>();
}

template <int N> [[nodiscard]] consteval auto make_literal() {
  return make_literal_impl<N>(
      std::make_index_sequence<static_cast<std::size_t>(N + 1)>{});
}

using static_degree20_t = decltype(make_literal<20>());
} // namespace polygnition_codegen

using static_degree20_t = polygnition_codegen::static_degree20_t;

extern "C" double polygnition_codegen_static_bridge_target(
    static_degree20_t p, double x);
