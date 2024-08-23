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
                    "value": metric_functions.dar(facet_a_scores, facet_d_scores, facet_a_trues, facets_d_trues),
                    "name": "Difference in Acceptance Rates (DAR)",
                    "description": "Measures the difference on the ratios of obserevd positive outcomes (TP) to the predicted positives (TP + FP) between the favored and disfavored facets"
                }
            }
        )

        return metrics


