from azureml.core.model import Model


model = Model.register(
    model_path = "./checkpoints",
    model_name = "transformap",
    workspace = "grab-map"
)