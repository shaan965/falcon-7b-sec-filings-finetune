# Idenfity the foundation model to fine-tune
model_id = "huggingface-llm-falcon-7b-bf16"

# The training data of SEC filing of Amazon has been pre-saved in the S3 bucket.
from sagemaker.jumpstart.utils import get_jumpstart_content_bucket

# Sample training data is available in this bucket
data_bucket = get_jumpstart_content_bucket(aws_region)
data_prefix = "training-datasets/sec_data"

training_dataset_s3_path = f"s3://{data_bucket}/{data_prefix}/train/"
validation_dataset_s3_path = f"s3://{data_bucket}/{data_prefix}/validation/"

#Prepare training parameters
from sagemaker import hyperparameters

my_hyperparameters = hyperparameters.retrieve_default(
    model_id=model_id, model_version=model_version
)

my_hyperparameters["epoch"] = "3"
my_hyperparameters["per_device_train_batch_size"] = "2"
my_hyperparameters["instruction_tuned"] = "False"
print(my_hyperparameters)
# Validate hyperparameters

hyperparameters.validate(
    model_id=model_id, model_version=model_version, hyperparameters=my_hyperparameters
)
# Starting training
from sagemaker.jumpstart.estimator import JumpStartEstimator

domain_adaptation_estimator = JumpStartEstimator(
    model_id=model_id,
    hyperparameters=my_hyperparameters,
    instance_type="ml.p3dn.24xlarge",
)
domain_adaptation_estimator.fit(
    {"train": training_dataset_s3_path, "validation": validation_dataset_s3_path}, logs=True
)
#Extract Training performance metrics. Performance metrics such as training loss and validation accuracy/loss can be accessed through cloudwatch while the training. We can also fetch these metrics and analyze them within the notebook

from sagemaker import TrainingJobAnalytics

training_job_name = domain_adaptation_estimator.latest_training_job.job_name

df = TrainingJobAnalytics(training_job_name=training_job_name).dataframe()
df.head(10)
# Deploying inference endpoints
# We deploy the domain-adaptation fine-tuned and pretrained models separately, and compare their performances.
# We first deploy the domain-adaptation fine-tuned model.

domain_adaptation_predictor = domain_adaptation_estimator.deploy()
#Next, we deploy the pre-trained huggingface-llm-falcon-7b-bf16.

my_model = JumpStartModel(model_id=model_id)
pretrained_predictor = my_model.deploy()

# Running inference queries and compare model performances
parameters = {
    "max_new_tokens": 300,
    "top_k": 50,
    "top_p": 0.8,
    "do_sample": True,
    "temperature": 1,
}


def generate_response(endpoint_name, text):
    payload = {"inputs": f"{text}:", "parameters": parameters}
    query_response = query_endpoint_with_json_payload(
        json.dumps(payload).encode("utf-8"), endpoint_name=endpoint_name
    )
    generated_texts = parse_response(query_response)
    print(f"Response: {generated_texts}{newline}")
test_paragraph_domain_adaption = [
    "This Form 10-K report shows that",
    "We serve consumers through",
    "Our vision is",
]


for paragraph in test_paragraph_domain_adaption:
    print("-" * 80)
    print(paragraph)
    print("-" * 80)
    print(f"{bold}pre-trained{unbold}")
    generate_response(pretrained_predictor.endpoint_name, paragraph)
    print(f"{bold}fine-tuned{unbold}")
    generate_response(domain_adaptation_predictor.endpoint_name, paragraph)

# The fine-tuned model starts to generate responses that are more specific to the domain of fine-tuning data which is relating to SEC report of Amazon.

# Clean up the endpoint
# Delete the SageMaker endpoint
pretrained_predictor.delete_model()
pretrained_predictor.delete_endpoint()
domain_adaptation_predictor.delete_model()
domain_adaptation_predictor.delete_endpoint()