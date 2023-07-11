from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["Model Input Table", "params:split_params"],
                outputs=[
                    "X_Train",
                    "X_Valid",
                    "X_Test",
                    "Y_Train",
                    "Y_Valid",
                    "Y_Test",
                ],
                name="Split_Data_Node",
            ),
            node(
                func=train_model,
                inputs=["X_Train", "X_Valid", "Y_Train", "Y_Valid"],
                outputs="Model",
                name="Train_Model_Node",
            ),
            node(
                func=evaluate_model,
                inputs=["Model", "X_Test", "Y_Test"],
                outputs=None,
                name="Evaluate_Model_Node",
            ),
        ]
    )
