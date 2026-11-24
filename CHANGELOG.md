# Changelog

- (24/11/2026) **Unreleased** Local polar metrics and cos weighting (follow-up to PR #8)
  - Modifications
    - calculate_localisation_error returns the local polar metrics (pol_local_*) by default instead of pol_* as quadrant errors inflate the global polar spread. For complete reporting, local polar metrics should be accompanied by quadrant_error
    - Polar errors and quadrant error are weighted with cos instead of cos^2 of the lateral target (r >= 0.95 in our data), see weighted_method
    - polar_error_weight returns (weight, used)
    - This breaks backward compatibility, use pol_local=False and weighted_method='cos2' for the old behaviour
