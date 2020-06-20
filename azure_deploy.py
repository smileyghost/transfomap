import azureml.core
from azureml.core.workspace import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
from azureml.core.webservice import Webservice
from azureml.core.model import Model
from azureml.core.environment import Environment

ws = Workspace.from_config()

compute_name = 'onizuka'
try:
    compute_target = ComputeTarget(workspace=ws, name=compute_name)
    print('Found existing compute target.')
except:
    print('not found')

#print(compute_target.get_status().serialize())

myenv = Environment.from_existing_conda_environment("pytorch", "pytorch")
inference_config = InferenceConfig(entry_script="transformap_inference.py", environment=myenv, source_directory='./')
aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                               memory_gb=1, 
                                               tags={'data': 'image and query',  'method':'TransfoMap', 'framework':'pytorch'},
                                               description='Estimate time travel')
# Select
model = Model(model_name = 'onizuka-eta', model_path = 'checkpoints/checkpoint.pth', workspace=ws, id='onizuka-eta:2')
# Register
# model = Model.register(model_name = 'onizuka-eta', model_path = 'checkpoints/checkpoint.pth', workspace=ws)
print(model.name, model.id, model.version, sep = '\t')

service = Model.deploy(workspace=ws, 
                           name='aci-birds', 
                           models=[model], 
                           inference_config=inference_config, 
                           deployment_config=aciconfig)
service.wait_for_deployment(True)
print(service.state)
service.get_logs()
print(service.scoring_uri)