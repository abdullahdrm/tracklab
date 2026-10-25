# Analytics resolution audit

The analytics configuration now declares whether line endpoints are in source
pixels or normalized coordinates. Normalized endpoints are validated in
`[0, 1]`, resolved once against the source frame dimensions, and passed to the
line counter and renderer as the same immutable geometry. Pixel-space
configurations retain their legacy behavior.

The implementation is independent of the MOT17 detector-comparison and
fine-tuning paths: those evaluations do not run production video analytics or
the line counter. The recorded fine-tuning regression therefore remains a
model/comparison result, not an analytics-resolution artifact.

Regression coverage includes legacy pixels, normalized coordinates at multiple
resolutions, inclusive boundaries, finite/range validation, directional
crossings, configuration reuse, and shared counter/renderer geometry.
