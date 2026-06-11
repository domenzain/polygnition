#pragma once

namespace polygnition::target {
enum class profile : unsigned {
  generic = 0,
  generic_x86_64_fma = generic,
  x86_intel_coffee_lake = 1,
  x86_intel_icelake_avx512 = 2,
  x86_amd_zen3 = 3,
};

#if defined(POLYGNITION_TARGET_PROFILE)
static constexpr auto configured =
    static_cast<profile>(POLYGNITION_TARGET_PROFILE);
#else
static constexpr auto configured = profile::generic;
#endif
} // namespace polygnition::target
