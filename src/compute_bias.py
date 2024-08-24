from bias_metrics.pre_training_bias import PreTrainingBias
from bias_metrics.post_training_bias import PostTrainingBias
import pandas as pd
from typing import Union, Optional
import numpy as np
from enum import Enum


class ModelType(Enum):
    LINEAR_REGRESSION = "LinearRegression"
    LOGISTIC_REGRESSION = "LogisitcRegression"
    BINARY_CLASSIFICATION = "BinaryClassification"
    MULTI_CLASS_CLASSIFICATION = "MultiClassClassification"

class FindBias:
    @staticmethod
    def compute_post_training_bias(
            df: pd.DataFrame,
            model_type: str,
            label_column_name: str,
            target_column_name: str,
            feature_name: str,
            feature_threshold: Optional[float] = None,
            decision_threshold: Optional[float] = None,
            target_label: Optional[str] = None,
            advantaged_class_value: Union[str, int, float] = None,
            prediction_threshold: Optional[float] = None
        ) -> dict:
        if model_type == ModelType.LINEAR_REGRESSION.value:
            if not prediction_threshold:
                raise Exception("Predcition threshold must be set for LinearRegression")
            df[target_column_name] = np.where(df[target_column_name] >= prediction_threshold, 1, 0)
            df[label_column_name] = np.where(df[label_column_name].to_numpy() >= prediction_threshold, 1, 0)

        elif model_type == ModelType.LOGISTIC_REGRESSION.value:
            if not decison_threshold:
                raise Exception("Decision threshold must be set for LogisticRegression")
            df[target_column_name] = np.where(df[target_column_name] >= decision_threshold, 1, 0)
            df[label_column_name] = np.where(df[label_column_name] >= decision_threshold, 1, 0)
        else:
            if not target_label:
                raise Exception(f"Target_label must be set for {ModelType.BINARY_CLASSIFICATION.value} and {ModelType.MULTI_CLASS_CLASSIFICATION.value}")


        if feature_threshold:
            df_a = df[df[feature_name] >= feature_threshold]
            df_d = df[df[feature_name] < feature_threshold]

            facet_a_trues = df_a[label_column_name].to_numpy()
            facet_a_scores = df_a[target_column_name].to_numpy()
            facet_d_trues = df_d[label_column_name].to_numpy()
            facet_d_scores = df_d[target_column_name].to_numpy()

        else:
            df_a = df[df[feature_name] == advantaged_class_value]
            df_d = df[df[feature_name] != advantaged_class_value]

            facet_a_trues = df_a[label_column_name].to_numpy()
            facet_a_scores = df_a[target_column_name].to_numpy()
            facet_d_trues = df_d[label_column_name].to_numpy()
            facet_d_scores = df_d[target_column_name].to_numpy()

        y_true = df[label_column_name].to_numpy()
        y_pred = df[target_column_name].to_numpy()

        metric_functions = PostTrainingBias()

        metrics = dict()

        metrics.update(
            {
                "ddpl": {
                    "value": metric_functions(facet_a_scores, facet_d_scores, facet_a_trues, facet_d_trues),
                    "name": "Difference in Positive Proportions in Predicted Labels (DDPL)",
                    "description": "Measures the difference in the proportion of positive predictions between the favored facet a and the disfavored facet d"
                }
            }
        )

        metrics.update(
            {
                "di": {
                    "value": metric_functions.di(facet_a_scores, facet_d_scores, facet_a_trues, facet_d_trues),
                    "name": "Disparate Impact",
                    "description": "Measures the ratio of propirtions of the predicted labels for the favored facet and the disfavored facet"
                }
            }
        )

        metrics.update(
            {
                "ad": {
                    "value": metric_functions.ad(facet_a_scores, facet_d_scores, facet_a_trues, facet_d_trues),
                    "name": "Accuracy Difference (AD)",
                    "description": "Measures the difference between the prediction accuracy for the favored and disfavored facets"
                }
            }
        )

        metrics.update(
            {
                "rd": {
                    "value": metric_functions.rd(facet_a_scores, facet_d_scores, facet_a_trues, facet_d_trues),
                    "name": "Recall Difference (RD)",
                    "description": "The difference in the recall metirc from facet a to facet d"
                }
            }
        )

        metrics.update(
            {
                "cdacc": {
                    "value": metric_functions.cdacc(facet_a_scores, facet_d_scores, facet_a_trues, facet_d_trues),
                    "name": "Difference in Conditional Acceptance (DCAcc)",
                    "description": " Compares the observed labels to the labels predicted by the model. Asses whether this is the same across facets for predicted positive outcomes"
                }
            }
        )

        metrics.update(
            {
                "dar": {
                    "value": metric_functions.dar(facet_a_scores, facet_d_scores, facet_a_trues, facet_d_trues),
                    "name": "Difference in Acceptance Rates (DAR)",
                    "description": "Measures the difference on the ratios of obserevd positive outcomes (TP) to the predicted positives (TP + FP) between the favored and disfavored facets"
                }
            }
        )

        metrics.update(
            {
                "sd": {
                    "value": metric_functions.sd(facet_a_scores, facet_d_scores, facet_a_trues, facet_d_trues),
                    "name": "Specifigity Difference (SD)",
                    "description": "Compares the specificity of the model between the favored and disfavored facets"
                }
            }
        )

        metrics.update(
            {
                "dcr": {
                    "value": metric_functins.dcr(facet_a_scores, facet_d_scores, facet_a_trues, facet_d_trues),
                    "name": "Difference is Conditional Rejection (DCR)",
                    "description": "Compares the observed labels to the labels predicted by a model and assessses whether this is the same accross facets for negative outcomes (rejection)"
                }
            }
        )


        metric.update(
            {
                "drr": {
                    "value": metric_functions.drr(facet_a_scores, facet_d_scores, facet_a_trues, facet_d_trues),
                    "name": "Difference in Rejection Rates",
                    "description": "Measures the difference in the ratios of the observed negative outcomes (TN) to the predicted negative outcomes between the favored and disfavored facets"
                }
            }
        )

        metrics.update(
            {
                "te": {
                    "value": metric_functions.te(facet_a_scores, facet_d_scores, facet_a_trues, facet_d_trues),
                    "name": "Treatment Equity (TE)",
                    "description": "Measures the difference in the ratio of false positives to false negatives between the favored and disfavored facets"
                }
            }
        )

        metrics.update(
            {
                "ge": {
                    "value": metric_functions.ge(y_pred, y_true),
                    "name": "Generalized Entropy (GE)",
                    "description": "Measures the inequality in benefits b assigned to each input by the model prediction. Assesses whether one facet benefits from positive predictions"
                }
            }
        )
        return {
            "feature": feature_name,
            "metrics": metrics
        }


    @staticmethod
    def compute_pre_training_bias(
            df: pd.DataFrame,
            feature_name: str,
            label_column_name: str,
            model_type: str,
            feature_threshold: Optional[float] = None,
            label_threshold: Optional[float] = None,
            advantaged_class_value: Optional[str] = None
        ) -> dict:
        if not all([
                feature_threshold or advantaged_class_value,
                not (feature_threshold and advantaged_class_value),
                feature_name not in df.columns.to_list(),
                model_type in [model_type.value for model_type in ModelType]]
            ):
            raise Exception("One of feature_threshold or advantaged_class_value is required")

        if label_threshold:
            assert(np.issubdtype(df[label_column_name].dtype, np.integer) or np.issubdtype(df[label_column_name].dtype, np.float)), "label_threhold is not valid for label dtype: {np.integer, np.float}"
            df[label_column_name] = np.where(df[label_column_name] >= label_threshold, 1, 0)

        if feature_threshold:
            assert(np.issubdtype(df[feature_name].dtype, np.integer) or np.issubdtype(df[feature_name].dtype, np.float)), "label_threhold is not valid for label dtype: {np.integer, np.float}"
            df[feature_name] = np.where(df[feature_name] >= feature_threshold, 1, 0)
            facet_a_trues = df[df[feature_name] == 1][label_column_name].to_numpy()
            facet_d_trues = df[df[feature_name] == 0][label_column_name].to_numpy()

        else:
            facet_a_trues = df[df[feature_name] == advantaged_class_value][label_column_name].to_numpy()
            facet_d_trues = df[df[feature_name] != advantaged_class_value][label_column_name].to_numpy()

        assert (facet_a_trues.size > 0 and facet_d_trues.size > 0), "All examples belong to a single facet"
        metric_functions = PreTrainingBias()

        metrics = dict()

        metrics.update(
            {
                "ci": metric_functions.ci(facet_a_trues, facet_d_trues),
                "name": "Class Imbalance",
                "description": "This measures a potential imbalance in the members present in the facet"
            }
        )

        metrics.update(
            {
                "dpl": {
                    "value": metric_functions.dpl(facet_a_trues, facet_d_trues),
                    "name": "Difference in Proportion of Labels (DPL)",
                    "description": "Measures the imbalance of positive outcomes between different facet values"
                }
            }
        )

        metrics.update(
            {
                "kll": {
                    "value": metric_functions.kll(facet_a_trues, facet_d_trues),
                    "name": "Kullback-Leibler Divergence",
                    "description": "Measures how much the outcome of distributions of different facets diverge from each other entropically"
                }
            }
        )

        metrics.update(
            {
                "js": {
                    "value": metric_functions.js(facet_a_trues, facet_d_trues),
                    "name": "Jensen-Shannon Divergence (JS)",
                    "description": "Simialr to KL. Indicates how the distribution of outcomes diverge"
                }
            }
        )

        metrics.update(
            {
                "lp-norm": {
                    "value": metric_functions.lp_norm(facet_a_trues, facet_d_trues),
                    "name": "LP-Norm",
                    "Description": "Measures p-norm difference between distinct demographic distributions of the outcome assocaited with different facets in a dataset. Uses 2-norm (Euclidean Distance)"
                }
            }
        )

        metrics.update(
            {
                "tvd": {
                    "value": metric_functions.tvd(facet_a_trues, facet_d_trues),
                    "name": "Total Variation Difference (TVD)",
                    "description": "Measures half of the 1-norm difference between distinct distributions of the outcomes associated with different facets in the dataset"
                }
            }
        )

        metrics.update(
            {
                "ks": {
                    "value": metric_functions.ks(facet_a_trues, facet_d_trues),
                    "name": "Kolmogorov-Smirnov (KS)",
                    "description": "Measures the maximum divergence between outcomes in distributions fro different facets in the dataset"
                }
            }
        )

        return {
            "feature": feature_name,
            "metrics": metrics
        }
