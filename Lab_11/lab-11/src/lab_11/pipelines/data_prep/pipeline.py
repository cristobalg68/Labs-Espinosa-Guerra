from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_model_input_table,
    get_data,
    preprocess_companies,
    preprocess_shuttles,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_data,
                inputs=None,
                outputs=["Companies", "Shuttles", "Reviews"],
                name="Get_Data_Node",
            ),
            node(
                func=preprocess_companies,
                inputs="Companies",
                outputs="Preprocessed Companies",
                name="Preprocess_Companies_Node",
            ),
            node(
                func=preprocess_shuttles,
                inputs="Shuttles",
                outputs="Preprocessed Shuttles",
                name="Preprocess_Shuttles_Node",
            ),
            node(
                func=create_model_input_table,
                inputs=["Preprocessed Shuttles", "Preprocessed Companies", "Reviews"],
                outputs="Model Input Table",
                name="Create_Model_Input_Table_Node",
            ),
        ]
    )
