from .wan_i2v import (
    WanHiF4PreprocessConfig, WanNativeModuleState, WanPreparedHiF4Linear,
    WanSmoothedWeightRecord, native_hif4_activation_qdq, native_hif4_weight_qdq,
    replace_wan_expert_with_preprocessed_hif4, select_wan_hif4_linears,
    setup_wan_i2v_hif4_preprocess, load_wan_activation_stats,
    load_wan_curvature_stats, load_wan_smoothed_weights,
)
