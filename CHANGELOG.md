# Changelog

## Unreleased

- The polar error weight (`polar_weight`) is now cos(lateral target) instead of cos²(lateral target). The previous weight is available with `polar_error_weight(df, method='cos2')` (see PR #8). On our data, the cos and cos² versions correlate at Pearson r >= 0.95.
- `polar_error_weight(df, method)` returns `(weight, used)`, `used` being a boolean mask of the responses that are used. New `method='split'`: constant weight for the targets within 30 degrees of the median plane, the others are not used (Middlebrooks 1999).
- `calculate_quadrant_error(df, weighted_method='cos')` uses all responses, weighted with `polar_error_weight`. This is the new default (also for `quadrant_error` of `calculate_localisation_error`); the Middlebrooks quadrant error is `weighted_method='split'`.
- `calculate_localisation_error(df, *grouping_vars, weighted_method='cos')` weights the polar errors and the quadrant error with `polar_error_weight` instead of the `polar_weight` column.
- Polar accuracy and precision only use responses with |polar error| <= 90 degrees (before weighting), as Middlebrooks (1999): front-back confusions make the errors bimodal, and their mean and spread do not describe that. The error is wrapped before weighting.
