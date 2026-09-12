from typing import Dict, Any

# Empirical reference scales derived from dataset EDA percentiles (feature_stats.csv)
NORM_MAX_WIDTH = 18.0   # Width is primary risk indicator in visual inspection
NORM_LENGTH = 700.0     # Crack length reflects extension across the member
NORM_AREA = 5.0         # Affected surface percentage

# Weight distribution (sum = 1.0)
WEIGHT_WIDTH = 0.45     # In structural inspection guidelines (width is paramount)
WEIGHT_LENGTH = 0.30    # Extent of visible propagation
WEIGHT_AREA = 0.25      # Surface impact density

# Calibration thresholds based on empirical 33rd and 66th percentile score boundaries
THRESHOLD_LOW = 0.40
THRESHOLD_MEDIUM = 0.70

def compute_visual_severity(features: Dict[str, Any]) -> Dict[str, Any]:
    "inspection severity rating from robust rules"
    if not features.get('has_crack', False) or features.get('crack_pixel_count', 0) == 0:
        return {
            'severity': 'None',
            'score': 0.0,
            'color': '#64748b',
            'sub_scores': {'width_score': 0.0, 'length_score': 0.0, 'area_score': 0.0},
            'rationale': 'No crack pixels were detected in the processed image.',
            'recommendation': 'No visible cracks detected. Continue routine periodic monitoring.'
        }

    max_w = float(features.get('max_width_px', 0.0))
    length = float(features.get('length_px', 0.0))
    area = float(features.get('affected_area_pct', 0.0))

    norm_w = min(1.0, max_w / NORM_MAX_WIDTH)
    norm_l = min(1.0, length / NORM_LENGTH)
    norm_a = min(1.0, area / NORM_AREA)

    w_w = WEIGHT_WIDTH * norm_w
    w_l = WEIGHT_LENGTH * norm_l
    w_a = WEIGHT_AREA * norm_a
    total_score = round(w_w + w_l + w_a, 3)

    if total_score < THRESHOLD_LOW:
        severity = 'Low'
        color = '#22c55e'
        rec = 'Hairline or minor superficial crack characteristics. Visual characteristics suggest minimal immediate surface distress; continue regular scheduled monitoring.'
    elif total_score < THRESHOLD_MEDIUM:
        severity = 'Medium'
        color = '#eab308'
        rec = 'Moderate crack characteristics observed in length and width. Recommended for on-site visual re-inspection within the standard maintenance cycle to check for active propagation.'
    else:
        severity = 'High'
        color = '#ef4444'
        rec = 'Significant crack dimensions (wide opening or extended path). Priority on-site physical inspection by a qualified structural engineer is recommended to assess depth, causes, and structural context.'


    rationale = (
        f"Assigned Visual Severity: {severity} (Score: {total_score:.2f} / 1.00)\n"
        f"Score Calculation Breakdown:\n"
        f"  - Max Width: {max_w:.1f} px (Normalized: {norm_w:.2f}, Weight: {WEIGHT_WIDTH}) -> +{w_w:.3f}\n"
        f"  - Crack Length: {length:.0f} px (Normalized: {norm_l:.2f}, Weight: {WEIGHT_LENGTH}) -> +{w_l:.3f}\n"
        f"  - Affected Area: {area:.2f}% (Normalized: {norm_a:.2f}, Weight: {WEIGHT_AREA}) -> +{w_a:.3f}\n"
        f"Threshold Rules:\n"
        f"  - Score < {THRESHOLD_LOW:.2f} -> Low\n"
        f"  - {THRESHOLD_LOW:.2f} <= Score < {THRESHOLD_MEDIUM:.2f} -> Medium\n"
        f"  - Score >= {THRESHOLD_MEDIUM:.2f} -> High\n"
        f"Disclaimer: This is an automated visual assessment of 2D surface imagery only, not a structural safety calculation."
    )

    return {
        'severity': severity,
        'score': total_score,
        'color': color,
        'thresholds': {'T1_low': THRESHOLD_LOW, 'T2_medium': THRESHOLD_MEDIUM},
        'sub_scores': {
            'width_score': round(w_w, 3),
            'length_score': round(w_l, 3),
            'area_score': round(w_a, 3)
        },
        'rationale': rationale,
        'recommendation': rec
    }

if __name__ == '__main__':
    test_feat = {'has_crack': True, 'crack_pixel_count': 2500, 'max_width_px': 14.5, 'length_px': 620, 'affected_area_pct': 3.8}
    res = compute_visual_severity(test_feat)
    print(res['rationale'])
    print('\nRecommendation:', res['recommendation'])
