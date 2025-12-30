"""
Statistical Power Analysis Example

Demonstrates sample size calculations and power analysis.
"""

import sys
sys.path.append('..')

from assay_design_calculator import StatisticalPowerAnalysis

print("""
╔═══════════════════════════════════════════════════════════════════╗
║     STATISTICAL POWER ANALYSIS                                    ║
╚═══════════════════════════════════════════════════════════════════╝
""")

power_calc = StatisticalPowerAnalysis()

# Example 1: Calculate required sample size
print("\n=== Example 1: Sample Size Calculation ===\n")

effect_sizes = [0.2, 0.5, 0.8, 1.0]
print("Required sample size for different effect sizes:")
print("(α = 0.05, Power = 80%, two-sided test)\n")

for es in effect_sizes:
    n = power_calc.calculate_sample_size(
        effect_size=es,
        alpha=0.05,
        power=0.8,
        alternative='two-sided'
    )
    
    effect_desc = {
        0.2: "Small",
        0.5: "Medium",
        0.8: "Large",
        1.0: "Very Large"
    }
    
    print(f"  Effect Size {es} ({effect_desc.get(es, 'Custom')}): n = {n} per group")

# Example 2: Calculate power for given sample size
print("\n\n=== Example 2: Power Calculation ===\n")

sample_sizes = [10, 20, 30, 50, 100]
effect_size = 0.5

print(f"Statistical power for effect size = {effect_size} (Medium)")
print("(α = 0.05, two-sided test)\n")

for n in sample_sizes:
    power = power_calc.calculate_power(
        n=n,
        effect_size=effect_size,
        alpha=0.05,
        alternative='two-sided'
    )
    
    print(f"  n = {n:3d} per group: Power = {power:.1%}")

# Example 3: Visualize power curves
print("\n\n=== Example 3: Power Curves ===\n")

print("Generating power curve plot...")
print("Shows how sample size requirements change with effect size")
print("for different power levels (70%, 80%, 90%, 95%)")

power_calc.plot_power_curve(save_path='results/power_curve.png')

# Example 4: Practical scenario
print("\n\n=== Example 4: Practical Scenario ===\n")

print("Scenario: Comparing two treatment groups")
print("-" * 60)

# Expected effect
expected_mean_diff = 10  # units
pooled_std = 15  # units
effect_size = expected_mean_diff / pooled_std

print(f"Expected mean difference: {expected_mean_diff} units")
print(f"Expected pooled std dev: {pooled_std} units")
print(f"Cohen's d (effect size): {effect_size:.3f}")

# Calculate required n
n_80 = power_calc.calculate_sample_size(effect_size, power=0.8)
n_90 = power_calc.calculate_sample_size(effect_size, power=0.9)

print(f"\nRequired sample sizes:")
print(f"  For 80% power: {n_80} per group (total: {n_80*2})")
print(f"  For 90% power: {n_90} per group (total: {n_90*2})")

# What power do we have with limited budget?
budget_n = 25  # Can only afford 25 per group
actual_power = power_calc.calculate_power(budget_n, effect_size)

print(f"\nWith budget constraint of n = {budget_n} per group:")
print(f"  Actual power: {actual_power:.1%}")

if actual_power < 0.8:
    print(f"  ⚠️  Warning: Power < 80% - may miss real effects!")
    n_needed = n_80 - budget_n
    print(f"  Need {n_needed} more samples per group for 80% power")
else:
    print(f"  ✅ Good! Power ≥ 80%")

# Example 5: One-sided vs Two-sided tests
print("\n\n=== Example 5: One-sided vs Two-sided Tests ===\n")

effect_size = 0.5
power = 0.8

n_two_sided = power_calc.calculate_sample_size(
    effect_size, power=power, alternative='two-sided'
)
n_one_sided = power_calc.calculate_sample_size(
    effect_size, power=power, alternative='one-sided'
)

print(f"Effect size: {effect_size}, Power: {power:.0%}")
print(f"  Two-sided test: n = {n_two_sided} per group")
print(f"  One-sided test: n = {n_one_sided} per group")
print(f"  Savings with one-sided: {n_two_sided - n_one_sided} per group")
print("\n  Note: Use one-sided tests only when you have strong")
print("  a priori reason to expect effect in one direction!")

print("\n" + "="*70)
print("✅ Power Analysis Complete!")
print("📁 Check 'results/' folder for:")
print("   - power_curve.png")
print("="*70 + "\n")