import features as feat
import features_visual as feat_v
import features_electrical as feat_e
import features_correlations as feat_c

features_standard_metadata = [
    feat.Feature_load_manual_labels,
    feat.Feature_load_dataset_metadata,
]

features_standard_electrical = [
    # # electrical
    feat_e.Feature_load_spike_times, feat_e.Feature_spikes_basic,
    feat_e.Feature_generate_acf_from_spikes,
    feat_e.Feature_load_ei,
    feat_e.Feature_spike_waveform,
    feat_e.Feature_ei_correlation_data,
    feat_e.Feature_ei_select_electrodes, feat_e.Feature_ei_map,
    # feat_e.Feature_ei_profile,
    # feat_e.Feature_stimulus_TTL,
]

features_standard_visual = [
    # feat_v.Feature_sta_vision_fit,
    feat_v.Feature_load_sta,
    feat_v.Feature_detect_grayscale_STA,
    feat_v.Feature_sta_basic,
    feat_v.Feature_wn_params,

    feat_v.Feature_sigstix,
    feat_v.Feature_primary_colors,
    feat_v.Feature_timecourses,

    feat_v.Feature_center_of_mass,
    feat_v.Feature_rf_map,
    # feat_v.Feature_rf_advanced_properties,
    # feat_v.Feature_rf_histogram,
    feat_v.Feature_edge_detection,
    feat_v.Feature_rf_boundary,
    feat_v.Feature_rf_convex_hull,
]

features_standard_deduplication_only = [
    feat.Feature_load_manual_labels,
    feat.Feature_load_dataset_metadata,
    feat_e.Feature_load_spike_times,
    feat_e.Feature_spikes_basic,
    feat_e.Feature_load_ei,
    feat_e.Feature_spike_waveform,
    feat_e.Feature_ei_correlation_data
]

features_standard_precombined = [
    feat.Feature_load_manual_labels,
    feat.Feature_load_dataset_metadata,
    feat_e.Feature_load_spike_times,
    feat_e.Feature_spikes_basic,
    feat_e.Feature_generate_acf_from_spikes,
    feat_e.Feature_load_ei,
    feat_e.Feature_spike_waveform,
    feat_e.Feature_ei_correlation_data,
    feat_v.Feature_load_sta,
    feat_v.Feature_wn_params,
]
features_standard_postcombined = [
    feat_v.Feature_detect_grayscale_STA,
    feat_v.Feature_sta_basic,
    feat_v.Feature_sigstix,
    feat_v.Feature_primary_colors,
    feat_v.Feature_timecourses,
    feat_v.Feature_rf_map,
    feat_v.Feature_center_of_mass,
    # feat_v.Feature_rf_advanced_properties,
    # feat_v.Feature_rf_histogram,
    feat_v.Feature_edge_detection,

    # feat_e.Feature_ei_correlation_data,
    feat_e.Feature_ei_select_electrodes, feat_e.Feature_ei_map,
    feat_e.Feature_ei_profile,
    feat_e.Feature_spike_waveform,
]

features_standard_correlation = [
    feat_c.Feature_rf_radii,
    feat_c.Feature_rf_overlaps,
    feat_c.Feature_cross_correlations_complete_fast,
]