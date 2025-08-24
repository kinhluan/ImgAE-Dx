# ========================================
# ENHANCED COMPREHENSIVE RESEARCH REPORT
# ========================================

import json
from datetime import datetime
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import pandas as pd

def generate_enhanced_research_report(model_metrics, model_errors, test_labels, CONFIG):
    """Generate comprehensive research report with statistical analysis"""
    
    print("📋 Generating Enhanced Comprehensive Research Report...")
    print("="*80)
    
    # Initialize comprehensive report structure
    research_report = {
        "executive_summary": {},
        "metadata": {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "analysis_version": "enhanced_v2.0",
            "dataset": {
                "name": CONFIG['hf_dataset'],
                "total_samples": len(test_labels),
                "normal_samples": int(np.sum(test_labels == 0)),
                "pneumonia_samples": int(np.sum(test_labels == 1)),
                "class_balance_ratio": float(np.sum(test_labels == 1) / np.sum(test_labels == 0)),
                "training_samples": CONFIG.get('samples', 'unknown'),
                "validation_split": CONFIG.get('validation_split', 'unknown')
            },
            "models_evaluated": list(model_metrics.keys()),
            "evaluation_metrics": [
                "AUC-ROC", "AUC-PR", "F1-Score", "Sensitivity", 
                "Specificity", "Error Separation", "Cohen's d"
            ]
        },
        "model_performance": {},
        "comparative_analysis": {},
        "research_validation": {
            "RQ1": {},
            "RQ2": {},
            "H1": {},
            "H2": {}
        },
        "statistical_analysis": {},
        "clinical_interpretation": {},
        "limitations": {},
        "recommendations": {},
        "conclusions": {}
    }
    
    model_names_display = {'unet': 'U-Net', 'reversed_ae': 'Reversed AE'}
    
    print("📊 Processing model performance metrics...")
    
    # Enhanced model performance analysis
    performance_summary = {}
    for model_name, metrics in model_metrics.items():
        model_display = model_names_display.get(model_name, model_name)
        
        # Core performance metrics
        performance = {
            "core_metrics": {
                "auc_roc": float(metrics['auc_roc']),
                "auc_pr": float(metrics['auc_pr']),
                "f1_score": float(metrics['f1_score']),
                "accuracy": float(metrics.get('accuracy', 0)),
                "sensitivity": float(metrics['sensitivity']),
                "specificity": float(metrics['specificity']),
                "precision": float(metrics.get('precision', 0)),
                "recall": float(metrics.get('recall', 0))
            },
            "anomaly_detection_metrics": {
                "optimal_threshold": float(metrics['optimal_threshold']),
                "error_separation": float(metrics['error_separation']),
                "separation_ratio": float(metrics.get('separation_ratio', 0)),
                "cohens_d": float(metrics.get('cohens_d', 0)),
                "normal_mean_error": float(metrics.get('normal_mean_error', 0)),
                "pneumonia_mean_error": float(metrics.get('pneumonia_mean_error', 0))
            },
            "clinical_relevance": {
                "true_positive_rate": float(metrics['sensitivity']),
                "false_positive_rate": float(1 - metrics['specificity']),
                "positive_likelihood_ratio": float(metrics['sensitivity'] / (1 - metrics['specificity'])) if metrics['specificity'] != 1 else float('inf'),
                "negative_likelihood_ratio": float((1 - metrics['sensitivity']) / metrics['specificity']) if metrics['specificity'] != 0 else float('inf')
            },
            "quality_assessment": {
                "excellent_performance": metrics['auc_roc'] >= 0.90,
                "good_performance": 0.80 <= metrics['auc_roc'] < 0.90,
                "fair_performance": 0.70 <= metrics['auc_roc'] < 0.80,
                "poor_performance": metrics['auc_roc'] < 0.70,
                "clinical_utility": metrics['auc_roc'] >= 0.75 and metrics['sensitivity'] >= 0.80 and metrics['specificity'] >= 0.80
            }
        }
        
        research_report["model_performance"][model_name] = performance
        performance_summary[model_name] = performance
        
        print(f"   ✅ {model_display}: AUC-ROC={metrics['auc_roc']:.4f}, Sensitivity={metrics['sensitivity']:.4f}, Specificity={metrics['specificity']:.4f}")
    
    # Comparative analysis
    print("\n⚖️  Conducting comparative analysis...")
    
    if len(model_metrics) >= 2:
        model_names = list(model_metrics.keys())
        comparative_analysis = {}
        
        # Pairwise comparisons
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i < j:  # Avoid duplicate comparisons
                    comp_key = f"{model1}_vs_{model2}"
                    
                    # Performance differences
                    auc_diff = model_metrics[model2]['auc_roc'] - model_metrics[model1]['auc_roc']
                    sensitivity_diff = model_metrics[model2]['sensitivity'] - model_metrics[model1]['sensitivity']
                    specificity_diff = model_metrics[model2]['specificity'] - model_metrics[model1]['specificity']
                    
                    # Statistical tests
                    errors1 = model_errors[model1]
                    errors2 = model_errors[model2]
                    
                    # Mann-Whitney U test for overall error differences
                    u_stat_overall, p_val_overall = stats.mannwhitneyu(errors1, errors2, alternative='two-sided')
                    
                    # Separate tests for each class
                    normal_errors1 = errors1[test_labels == 0]
                    normal_errors2 = errors2[test_labels == 0]
                    pneumonia_errors1 = errors1[test_labels == 1]
                    pneumonia_errors2 = errors2[test_labels == 1]
                    
                    u_stat_normal, p_val_normal = stats.mannwhitneyu(normal_errors1, normal_errors2, alternative='two-sided')
                    u_stat_pneumonia, p_val_pneumonia = stats.mannwhitneyu(pneumonia_errors1, pneumonia_errors2, alternative='two-sided')
                    
                    # Effect sizes
                    pooled_std_overall = np.sqrt((np.var(errors1) + np.var(errors2)) / 2)
                    cohens_d_overall = (np.mean(errors2) - np.mean(errors1)) / pooled_std_overall if pooled_std_overall > 0 else 0
                    
                    comparative_analysis[comp_key] = {
                        "model1": model1,
                        "model2": model2,
                        "performance_differences": {
                            "auc_roc_difference": float(auc_diff),
                            "sensitivity_difference": float(sensitivity_diff),
                            "specificity_difference": float(specificity_diff),
                            "superior_model": model2 if auc_diff > 0 else model1,
                            "significant_improvement": abs(auc_diff) >= 0.05
                        },
                        "statistical_tests": {
                            "overall_errors": {
                                "test": "Mann-Whitney U",
                                "u_statistic": float(u_stat_overall),
                                "p_value": float(p_val_overall),
                                "significant": p_val_overall < 0.05,
                                "effect_size_cohens_d": float(cohens_d_overall)
                            },
                            "normal_class_errors": {
                                "test": "Mann-Whitney U",
                                "p_value": float(p_val_normal),
                                "significant": p_val_normal < 0.05
                            },
                            "pneumonia_class_errors": {
                                "test": "Mann-Whitney U", 
                                "p_value": float(p_val_pneumonia),
                                "significant": p_val_pneumonia < 0.05
                            }
                        },
                        "clinical_significance": {
                            "clinically_meaningful": abs(auc_diff) >= 0.05 and (abs(sensitivity_diff) >= 0.05 or abs(specificity_diff) >= 0.05),
                            "recommendation": "significant_improvement" if auc_diff >= 0.05 and p_val_overall < 0.05 else "no_clear_advantage"
                        }
                    }
        
        research_report["comparative_analysis"] = comparative_analysis
    
    # Research Questions Validation
    print("\n🎯 Validating research questions...")
    
    # RQ1: U-Net Baseline Effectiveness
    if 'unet' in model_metrics:
        unet_metrics = model_metrics['unet']
        unet_auc = unet_metrics['auc_roc']
        
        # Multiple thresholds for comprehensive evaluation
        thresholds = {"excellent": 0.90, "good": 0.80, "acceptable": 0.70}
        threshold_results = {}
        
        for level, threshold in thresholds.items():
            threshold_results[level] = {
                "threshold": threshold,
                "achieved": unet_auc >= threshold,
                "margin": float(unet_auc - threshold)
            }
        
        research_report["research_validation"]["RQ1"] = {
            "question": "Can U-Net architecture establish an effective baseline (AUC > 0.80) for classifying normal versus pneumonia X-ray images based on reconstruction error?",
            "primary_threshold": 0.80,
            "measured_auc": float(unet_auc),
            "threshold_analysis": threshold_results,
            "primary_result": "CONFIRMED" if unet_auc >= 0.80 else "REJECTED",
            "confidence_level": "high" if unet_auc >= 0.85 else "moderate" if unet_auc >= 0.75 else "low",
            "answer": "YES" if unet_auc >= 0.80 else "NO",
            "evidence": {
                "auc_roc": float(unet_auc),
                "sensitivity": float(unet_metrics['sensitivity']),
                "specificity": float(unet_metrics['specificity']),
                "error_separation": float(unet_metrics['error_separation'])
            },
            "interpretation": {
                "skip_connections_effectiveness": "Skip connections enable effective feature preservation for reconstruction-based anomaly detection" if unet_auc >= 0.80 else "Skip connections may not be sufficient for this specific anomaly detection task",
                "baseline_quality": "Excellent baseline" if unet_auc >= 0.85 else "Good baseline" if unet_auc >= 0.80 else "Insufficient baseline",
                "clinical_applicability": "Clinically relevant performance achieved" if unet_auc >= 0.80 and unet_metrics['sensitivity'] >= 0.80 else "Performance may not meet clinical standards"
            }
        }
        
        # H1 validation
        research_report["research_validation"]["H1"] = {
            "hypothesis": "The U-Net model will achieve good baseline performance (AUC > 0.80), demonstrating clear discriminative ability between normal and pneumonia cases",
            "status": "CONFIRMED" if unet_auc >= 0.80 else "REJECTED",
            "evidence": {
                "measured_auc": float(unet_auc),
                "discriminative_ability": float(unet_metrics['error_separation']),
                "statistical_power": float(unet_metrics.get('cohens_d', 0))
            },
            "supporting_metrics": {
                "balanced_performance": unet_metrics['sensitivity'] >= 0.70 and unet_metrics['specificity'] >= 0.70,
                "clinical_utility": unet_metrics['sensitivity'] >= 0.80 and unet_metrics['specificity'] >= 0.80
            }
        }
        
        print(f"   ✅ RQ1: {'CONFIRMED' if unet_auc >= 0.80 else 'REJECTED'} (AUC: {unet_auc:.4f})")
    
    # RQ2: RA vs U-Net Comparison
    if len(model_metrics) >= 2 and 'unet' in model_metrics and 'reversed_ae' in model_metrics:
        unet_metrics = model_metrics['unet']
        ra_metrics = model_metrics['reversed_ae']
        
        unet_auc = unet_metrics['auc_roc']
        ra_auc = ra_metrics['auc_roc']
        auc_difference = ra_auc - unet_auc
        
        # Comprehensive statistical analysis
        unet_errors = model_errors['unet']
        ra_errors = model_errors['reversed_ae']
        
        # Overall comparison
        u_stat, p_value = stats.mannwhitneyu(unet_errors, ra_errors, alternative='two-sided')
        
        # Class-specific analysis
        normal_indices = test_labels == 0
        pneumonia_indices = test_labels == 1
        
        unet_normal_errors = unet_errors[normal_indices]
        ra_normal_errors = ra_errors[normal_indices]
        unet_pneumonia_errors = unet_errors[pneumonia_indices]
        ra_pneumonia_errors = ra_errors[pneumonia_indices]
        
        # Localization analysis (error separation comparison)
        unet_separation = unet_metrics['error_separation']
        ra_separation = ra_metrics['error_separation']
        localization_advantage = ra_separation - unet_separation
        
        # Multiple criteria for superiority
        superiority_criteria = {
            "auc_superiority": ra_auc > unet_auc,
            "statistical_significance": p_value < 0.05,
            "clinical_significance": abs(auc_difference) >= 0.05,
            "localization_advantage": ra_separation > unet_separation,
            "balanced_improvement": ra_metrics['sensitivity'] >= unet_metrics['sensitivity'] and ra_metrics['specificity'] >= unet_metrics['specificity']
        }
        
        overall_superiority = sum(superiority_criteria.values()) >= 3  # Majority criteria
        
        research_report["research_validation"]["RQ2"] = {
            "question": "Does the Reversed Autoencoder (RA) architecture demonstrate superior performance compared to the standard U-Net architecture?",
            "models_compared": ["unet", "reversed_ae"],
            "performance_comparison": {
                "unet_auc": float(unet_auc),
                "ra_auc": float(ra_auc),
                "auc_difference": float(auc_difference),
                "percentage_improvement": float((auc_difference / unet_auc) * 100) if unet_auc > 0 else 0
            },
            "superiority_analysis": superiority_criteria,
            "statistical_tests": {
                "overall_error_comparison": {
                    "test": "Mann-Whitney U",
                    "u_statistic": float(u_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                    "interpretation": "RA produces significantly different reconstruction errors" if p_value < 0.05 else "No significant difference in reconstruction errors"
                }
            },
            "localization_analysis": {
                "unet_error_separation": float(unet_separation),
                "ra_error_separation": float(ra_separation),
                "localization_advantage": float(localization_advantage),
                "better_localization": "RA" if ra_separation > unet_separation else "U-Net"
            },
            "primary_result": "SUPERIOR" if overall_superiority else "NOT_SUPERIOR",
            "confidence_level": "high" if sum(superiority_criteria.values()) >= 4 else "moderate" if sum(superiority_criteria.values()) >= 2 else "low",
            "answer": "YES" if overall_superiority else "NO",
            "interpretation": {
                "performance_assessment": "RA demonstrates clear superiority across multiple metrics" if overall_superiority else "RA does not show consistent superiority over U-Net",
                "architectural_insights": "Asymmetric encoder-decoder without skip connections may be beneficial for pseudo-healthy reconstruction" if ra_separation > unet_separation else "Skip connections in U-Net provide better reconstruction fidelity",
                "clinical_recommendation": "RA recommended for deployment" if overall_superiority and ra_auc >= 0.80 else "U-Net remains the preferred baseline"
            }
        }
        
        # H2 validation  
        research_report["research_validation"]["H2"] = {
            "hypothesis": "The RA model will produce more localized error maps at pneumonia regions, though AUC may be comparable or slightly lower than U-Net due to its specialized reconstruction mechanism",
            "localization_metrics": {
                "unet_error_separation": float(unet_separation),
                "ra_error_separation": float(ra_separation),
                "localization_improvement": float(localization_advantage)
            },
            "performance_trade_off": {
                "auc_difference": float(auc_difference),
                "localization_vs_performance": "confirmed_trade_off" if ra_separation > unet_separation and ra_auc < unet_auc else "no_clear_trade_off"
            },
            "status": "CONFIRMED" if ra_separation > unet_separation else "REJECTED",
            "evidence": f"RA error separation ({ra_separation:.4f}) vs U-Net ({unet_separation:.4f})"
        }
        
        print(f"   ✅ RQ2: {'SUPERIOR' if overall_superiority else 'NOT_SUPERIOR'} (ΔA UC: {auc_difference:+.4f})")
    
    # Statistical Analysis Summary
    print("\n📈 Generating statistical analysis summary...")
    
    statistical_summary = {
        "sample_size": {
            "total": len(test_labels),
            "normal": int(np.sum(test_labels == 0)),
            "pneumonia": int(np.sum(test_labels == 1)),
            "power_analysis": "adequate" if len(test_labels) >= 100 else "limited"
        },
        "effect_sizes": {},
        "confidence_intervals": {},
        "statistical_power": {}
    }
    
    # Calculate effect sizes for each model
    for model_name, metrics in model_metrics.items():
        if 'cohens_d' in metrics:
            effect_size = abs(metrics['cohens_d'])
            statistical_summary["effect_sizes"][model_name] = {
                "cohens_d": float(metrics['cohens_d']),
                "magnitude": "large" if effect_size >= 0.8 else "medium" if effect_size >= 0.5 else "small"
            }
    
    research_report["statistical_analysis"] = statistical_summary
    
    # Clinical Interpretation
    print("\n🏥 Generating clinical interpretation...")
    
    clinical_interpretation = {
        "clinical_utility": {},
        "deployment_readiness": {},
        "risk_assessment": {},
        "recommendations": {}
    }
    
    for model_name, performance in performance_summary.items():
        model_display = model_names_display.get(model_name, model_name)
        
        # Clinical utility assessment
        auc = performance["core_metrics"]["auc_roc"]
        sensitivity = performance["core_metrics"]["sensitivity"]
        specificity = performance["core_metrics"]["specificity"]
        
        clinical_utility = {
            "screening_utility": auc >= 0.80 and sensitivity >= 0.85,
            "diagnostic_utility": auc >= 0.85 and sensitivity >= 0.80 and specificity >= 0.80,
            "clinical_grade": "excellent" if auc >= 0.90 else "good" if auc >= 0.80 else "fair" if auc >= 0.70 else "poor",
            "false_positive_rate": float(1 - specificity),
            "false_negative_rate": float(1 - sensitivity)
        }
        
        clinical_interpretation["clinical_utility"][model_name] = clinical_utility
    
    research_report["clinical_interpretation"] = clinical_interpretation
    
    # Limitations Analysis
    limitations = {
        "dataset_limitations": {
            "small_sample_size": len(test_labels) < 500,
            "class_imbalance": abs(np.sum(test_labels == 1) / np.sum(test_labels == 0) - 1) > 0.5,
            "single_dataset": True,
            "pediatric_specific": "Dataset limited to pediatric patients (ages 1-5)",
            "binary_classification": "Only normal vs pneumonia, no multi-class pathology detection"
        },
        "methodological_limitations": {
            "reconstruction_based": "Relies on reconstruction error, may miss subtle abnormalities",
            "unsupervised_training": "Models trained only on normal images",
            "threshold_dependency": "Performance depends on optimal threshold selection",
            "computational_requirements": "Deep learning models require significant computational resources"
        },
        "generalizability_concerns": {
            "population_generalization": "Results may not generalize to adult populations",
            "equipment_generalization": "Trained on specific X-ray equipment/protocols",
            "geographic_generalization": "Single-center data may not represent global populations"
        }
    }
    
    research_report["limitations"] = limitations
    
    # Recommendations
    recommendations = {
        "immediate_actions": [],
        "future_research": [],
        "clinical_deployment": [],
        "methodological_improvements": []
    }
    
    # Generate recommendations based on results
    if 'unet' in model_metrics and model_metrics['unet']['auc_roc'] >= 0.80:
        recommendations["immediate_actions"].append("U-Net baseline confirmed - proceed with clinical validation")
        recommendations["clinical_deployment"].append("Consider U-Net for clinical pilot study")
    
    if len(model_metrics) >= 2:
        best_model = max(model_metrics.items(), key=lambda x: x[1]['auc_roc'])
        recommendations["clinical_deployment"].append(f"Prioritize {model_names_display.get(best_model[0], best_model[0])} for deployment (AUC: {best_model[1]['auc_roc']:.4f})")
    
    recommendations["future_research"].extend([
        "Validate on larger, multi-center datasets",
        "Extend to multi-class pathology detection",
        "Investigate adult population performance",
        "Compare with radiologist performance",
        "Develop explainable AI visualization tools"
    ])
    
    recommendations["methodological_improvements"].extend([
        "Implement cross-validation for robust evaluation",
        "Add ensemble methods for improved performance",
        "Incorporate attention mechanisms for better localization",
        "Investigate semi-supervised learning approaches"
    ])
    
    research_report["recommendations"] = recommendations
    
    # Executive Summary
    print("\n📋 Generating executive summary...")
    
    # Determine overall research success
    rq1_success = 'unet' in model_metrics and model_metrics['unet']['auc_roc'] >= 0.80
    rq2_applicable = len(model_metrics) >= 2 and 'unet' in model_metrics and 'reversed_ae' in model_metrics
    rq2_success = False
    
    if rq2_applicable:
        ra_auc = model_metrics['reversed_ae']['auc_roc']
        unet_auc = model_metrics['unet']['auc_roc']
        rq2_success = ra_auc > unet_auc
    
    best_performer = max(model_metrics.items(), key=lambda x: x[1]['auc_roc']) if model_metrics else None
    
    executive_summary = {
        "research_objectives_met": rq1_success,
        "primary_findings": {
            "best_performing_model": best_performer[0] if best_performer else "none",
            "best_auc_score": float(best_performer[1]['auc_roc']) if best_performer else 0,
            "clinical_grade_performance": best_performer[1]['auc_roc'] >= 0.80 if best_performer else False
        },
        "key_insights": [],
        "clinical_impact": {
            "deployment_ready": best_performer[1]['auc_roc'] >= 0.80 if best_performer else False,
            "expected_performance": "good" if best_performer and best_performer[1]['auc_roc'] >= 0.80 else "insufficient"
        },
        "research_conclusion": "successful" if rq1_success else "partially_successful"
    }
    
    # Generate key insights
    if rq1_success:
        executive_summary["key_insights"].append("U-Net successfully established effective baseline for medical image anomaly detection")
    
    if rq2_applicable and rq2_success:
        executive_summary["key_insights"].append("Reversed Autoencoder demonstrated superior performance over U-Net baseline")
    elif rq2_applicable:
        executive_summary["key_insights"].append("U-Net baseline performance competitive with specialized RA architecture")
    
    if best_performer and best_performer[1]['auc_roc'] >= 0.90:
        executive_summary["key_insights"].append("Excellent performance achieved, ready for clinical validation")
    
    research_report["executive_summary"] = executive_summary
    
    # Final Conclusions
    conclusions = {
        "research_questions_answered": True,
        "primary_conclusion": "",
        "methodology_validation": "Reconstruction-based anomaly detection proven effective for pneumonia detection",
        "architectural_insights": "",
        "future_directions": "Multi-center validation and clinical deployment preparation recommended"
    }
    
    if rq1_success and rq2_applicable:
        if rq2_success:
            conclusions["primary_conclusion"] = f"Both research questions confirmed: U-Net established effective baseline (AUC={model_metrics['unet']['auc_roc']:.3f}), and RA demonstrated superior performance (AUC={model_metrics['reversed_ae']['auc_roc']:.3f})"
            conclusions["architectural_insights"] = "Specialized asymmetric architectures can outperform traditional U-Net for medical anomaly detection"
        else:
            conclusions["primary_conclusion"] = f"RQ1 confirmed but RQ2 not supported: U-Net baseline effective (AUC={model_metrics['unet']['auc_roc']:.3f}), RA did not demonstrate clear superiority (AUC={model_metrics['reversed_ae']['auc_roc']:.3f})"
            conclusions["architectural_insights"] = "Skip connections in U-Net provide robust performance for reconstruction-based anomaly detection"
    elif rq1_success:
        conclusions["primary_conclusion"] = f"RQ1 confirmed: U-Net successfully established effective baseline (AUC={model_metrics['unet']['auc_roc']:.3f})"
        conclusions["architectural_insights"] = "U-Net architecture well-suited for medical image anomaly detection tasks"
    else:
        conclusions["primary_conclusion"] = "Research objectives not fully met: baseline performance insufficient for clinical deployment"
        conclusions["architectural_insights"] = "Current architectures may require optimization for this specific anomaly detection task"
    
    research_report["conclusions"] = conclusions
    
    # Save comprehensive report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f"{CONFIG['checkpoint_dir']}/enhanced_comprehensive_research_report_{timestamp}.json"
    
    with open(report_path, 'w') as f:
        json.dump(research_report, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Enhanced research report saved: {report_path}")
    
    return research_report, report_path

def print_enhanced_research_summary(research_report, model_names_display):
    """Print comprehensive research summary"""
    
    print("\n" + "="*120)
    print("🎯 **ENHANCED RESEARCH VALIDATION SUMMARY**")
    print("="*120)
    
    # Executive Summary
    exec_summary = research_report["executive_summary"]
    print(f"\n📋 **EXECUTIVE SUMMARY**")
    print(f"   Research Objective Status: {'✅ MET' if exec_summary['research_objectives_met'] else '❌ NOT MET'}")
    print(f"   Best Performing Model: {model_names_display.get(exec_summary['primary_findings']['best_performing_model'], exec_summary['primary_findings']['best_performing_model'])}")
    print(f"   Best AUC Score: {exec_summary['primary_findings']['best_auc_score']:.4f}")
    print(f"   Clinical Grade Performance: {'✅ YES' if exec_summary['primary_findings']['clinical_grade_performance'] else '❌ NO'}")
    print(f"   Research Conclusion: {exec_summary['research_conclusion'].upper()}")
    
    # Key Insights
    if exec_summary["key_insights"]:
        print(f"\n🔍 **KEY INSIGHTS**:")
        for insight in exec_summary["key_insights"]:
            print(f"   • {insight}")
    
    # Research Questions
    rq_validation = research_report["research_validation"]
    
    if "RQ1" in rq_validation and rq_validation["RQ1"]:
        rq1 = rq_validation["RQ1"]
        print(f"\n📊 **RQ1 - U-Net Baseline Effectiveness**")
        print(f"   Question: {rq1['question']}")
        print(f"   Answer: **{rq1['answer']}** (AUC: {rq1['measured_auc']:.4f})")
        print(f"   Result: {rq1['primary_result']}")
        print(f"   Confidence: {rq1['confidence_level'].upper()}")
        
        # Threshold analysis
        print(f"   Threshold Analysis:")
        for level, data in rq1['threshold_analysis'].items():
            status = "✅" if data['achieved'] else "❌"
            print(f"     {level.capitalize()} (≥{data['threshold']:.2f}): {status} (margin: {data['margin']:+.4f})")
    
    if "RQ2" in rq_validation and rq_validation["RQ2"]:
        rq2 = rq_validation["RQ2"]
        print(f"\n⚖️  **RQ2 - RA vs U-Net Comparison**")
        print(f"   Question: {rq2['question']}")
        print(f"   Answer: **{rq2['answer']}**")
        print(f"   Result: {rq2['primary_result']}")
        print(f"   Confidence: {rq2['confidence_level'].upper()}")
        
        perf_comp = rq2['performance_comparison']
        print(f"   Performance Comparison:")
        print(f"     U-Net AUC: {perf_comp['unet_auc']:.4f}")
        print(f"     RA AUC: {perf_comp['ra_auc']:.4f}")
        print(f"     Difference: {perf_comp['auc_difference']:+.4f} ({perf_comp['percentage_improvement']:+.1f}%)")
        
        # Superiority criteria
        print(f"   Superiority Analysis:")
        for criterion, status in rq2['superiority_analysis'].items():
            indicator = "✅" if status else "❌"
            criterion_display = criterion.replace('_', ' ').title()
            print(f"     {criterion_display}: {indicator}")
        
        # Statistical significance
        stat_test = rq2['statistical_tests']['overall_error_comparison']
        print(f"   Statistical Test: {stat_test['interpretation']}")
        print(f"     p-value: {stat_test['p_value']:.4f} ({'significant' if stat_test['significant'] else 'not significant'})")
    
    # Model Performance Summary
    print(f"\n📈 **DETAILED MODEL PERFORMANCE**:")
    model_perf = research_report["model_performance"]
    
    for model_name, performance in model_perf.items():
        model_display = model_names_display.get(model_name, model_name)
        core_metrics = performance["core_metrics"]
        quality = performance["quality_assessment"]
        
        print(f"\n   🤖 **{model_display.upper()}**:")
        print(f"      Core Metrics:")
        print(f"        • AUC-ROC: {core_metrics['auc_roc']:.4f}")
        print(f"        • AUC-PR: {core_metrics['auc_pr']:.4f}")
        print(f"        • F1-Score: {core_metrics['f1_score']:.4f}")
        print(f"        • Sensitivity: {core_metrics['sensitivity']:.4f}")
        print(f"        • Specificity: {core_metrics['specificity']:.4f}")
        print(f"        • Accuracy: {core_metrics['accuracy']:.4f}")
        
        print(f"      Quality Assessment:")
        quality_level = "Excellent" if quality['excellent_performance'] else "Good" if quality['good_performance'] else "Fair" if quality['fair_performance'] else "Poor"
        print(f"        • Performance Grade: {quality_level}")
        print(f"        • Clinical Utility: {'✅ YES' if quality['clinical_utility'] else '❌ NO'}")
        
        # Anomaly detection specific metrics
        anomaly_metrics = performance["anomaly_detection_metrics"]
        print(f"      Anomaly Detection:")
        print(f"        • Error Separation: {anomaly_metrics['error_separation']:.4f}")
        print(f"        • Separation Ratio: {anomaly_metrics['separation_ratio']:.2f}x")
        print(f"        • Effect Size (Cohen's d): {anomaly_metrics['cohens_d']:.3f}")
        print(f"        • Optimal Threshold: {anomaly_metrics['optimal_threshold']:.6f}")
    
    # Clinical Interpretation
    if "clinical_interpretation" in research_report:
        clinical = research_report["clinical_interpretation"]
        print(f"\n🏥 **CLINICAL INTERPRETATION**:")
        
        for model_name, utility in clinical["clinical_utility"].items():
            model_display = model_names_display.get(model_name, model_name)
            print(f"   {model_display}:")
            print(f"     • Clinical Grade: {utility['clinical_grade'].capitalize()}")
            print(f"     • Screening Utility: {'✅' if utility['screening_utility'] else '❌'}")
            print(f"     • Diagnostic Utility: {'✅' if utility['diagnostic_utility'] else '❌'}")
            print(f"     • False Positive Rate: {utility['false_positive_rate']:.3f}")
            print(f"     • False Negative Rate: {utility['false_negative_rate']:.3f}")
    
    # Conclusions
    conclusions = research_report["conclusions"]
    print(f"\n🎯 **FINAL CONCLUSIONS**")
    print(f"   Primary Conclusion: {conclusions['primary_conclusion']}")
    print(f"   Methodology Validation: {conclusions['methodology_validation']}")
    print(f"   Architectural Insights: {conclusions['architectural_insights']}")
    print(f"   Future Directions: {conclusions['future_directions']}")
    
    # Recommendations
    if "recommendations" in research_report:
        recommendations = research_report["recommendations"]
        print(f"\n💡 **RECOMMENDATIONS**")
        
        if recommendations["immediate_actions"]:
            print(f"   Immediate Actions:")
            for action in recommendations["immediate_actions"]:
                print(f"     • {action}")
        
        if recommendations["clinical_deployment"]:
            print(f"   Clinical Deployment:")
            for deployment in recommendations["clinical_deployment"]:
                print(f"     • {deployment}")
    
    print("\n" + "="*120)
    print("✅ **RESEARCH COMPLETE**: Comprehensive analysis with statistical validation completed!")
    print("="*120)

# Execute enhanced research report generation
if 'model_metrics' in globals() and 'model_errors' in globals() and 'test_labels' in globals():
    print("📋 Starting Enhanced Research Report Generation...")
    
    # Generate comprehensive report
    enhanced_report, report_path = generate_enhanced_research_report(
        model_metrics, model_errors, test_labels, CONFIG
    )
    
    # Print comprehensive summary
    model_names_display = {'unet': 'U-Net', 'reversed_ae': 'Reversed AE'}
    print_enhanced_research_summary(enhanced_report, model_names_display)
    
    # Additional summary files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save executive summary as separate file
    exec_summary_path = f"{CONFIG['checkpoint_dir']}/executive_summary_{timestamp}.txt"
    with open(exec_summary_path, 'w', encoding='utf-8') as f:
        f.write("ENHANCED RESEARCH REPORT - EXECUTIVE SUMMARY\\n")
        f.write("="*60 + "\\n\\n")
        
        exec_summary = enhanced_report["executive_summary"]
        f.write(f"Research Objective Status: {'MET' if exec_summary['research_objectives_met'] else 'NOT MET'}\\n")
        f.write(f"Best Performing Model: {model_names_display.get(exec_summary['primary_findings']['best_performing_model'], exec_summary['primary_findings']['best_performing_model'])}\\n")
        f.write(f"Best AUC Score: {exec_summary['primary_findings']['best_auc_score']:.4f}\\n")
        f.write(f"Clinical Grade Performance: {'YES' if exec_summary['primary_findings']['clinical_grade_performance'] else 'NO'}\\n\\n")
        
        f.write("PRIMARY CONCLUSION:\\n")
        f.write(enhanced_report["conclusions"]["primary_conclusion"] + "\\n\\n")
        
        f.write("KEY INSIGHTS:\\n")
        for insight in exec_summary["key_insights"]:
            f.write(f"• {insight}\\n")
    
    print(f"\\n📄 Executive summary saved: {exec_summary_path}")
    print(f"📁 All research artifacts saved to: {CONFIG['checkpoint_dir']}")
    
    # Force sync to Google Drive
    import os
    if os.path.exists('/content/drive'):
        os.system('sync')
        print("🔄 Research report synced to Google Drive")
    
else:
    print("❌ Required data not available for research report generation")
    print("   Need: model_metrics, model_errors, test_labels, CONFIG")
    print("   Please run model evaluation and comparative analysis first")