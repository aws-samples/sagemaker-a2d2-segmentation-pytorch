# 2D Semantic Segmentation on the Audi A2D2 Dataset
In this notebook, we train a model on the 2D semantic segmentation annotations from the Audi A2D2 Dataset https://www.a2d2.audi/a2d2/en/dataset.html. The dataset can also be accessed from the the AWS Open Data Registry https://registry.opendata.aws/aev-a2d2/. 

We do the following: 

 1. We download the semantic segmentation dataset archive
 1. We inspect and describe the data
 1. We run local processing to produce a dataset manifest (list of all records), and split the data in training and validation sections. 
 1. We send the data to Amazon S3
 1. We create a PyTorch script training a DeepLabV3 model, that we test locally for few iterations
 1. We launch our training script on a remote, long-running machine with SageMaker Training API. 
 1. We show how to run bayesian parameter search to tune the metric of your choice (loss, accuracy, troughput...). To keep costs low, this is deactivated by default
 1. We open a model checkpoint (collected in parallel to training by SageMaker Training) to check prediction quality

The demo was created from a SageMaker ml.g4dn.16xlarge Notebook instance, with a Jupyter Kernel `conda_pytorch_latest_p37` (`torch 1.8.1+cu111`, `torchvision 0.9.1+cu111`). Feel free to use a different instance for the download and pre-processing step so that a GPU doesn't sit idle, and switch to a different instance type later. Note that the dataset download and extraction **do not run well on SageMaker Studio Notebooks**, whose storage is EFS based and struggles to handle the 80k+ files composing the dataset. Launching API calls (training job, tuning jobs) from Studio should run fine though.

**IMPORTANT NOTES**

* **This sample is written for single-GPU instances only. Using machines with more than 1 GPU or running the training code on more than 1 machines will not use all available hardware**

* **Running this demo necessitates at least 400 Gb of local storage space**

* **Running this demo on an ml.G4dn.16xlarge instance in region eu-west-1 takes approximately 50min of notebook uptime and approximately 12h of SageMaker Training job execution (excluding the bayesian parameter search, de-activated by default). This represents approximately 6 USD of notebook usage (if running on ml.g4dn.16xlarge) and 72 USD of training API**

* **This demo uses non-AWS, open-source libraries including PyTorch, PIL, matplotlib, Torchvision. Use appropriate due diligence to verify if that use fits the software standards and compliance rules in place at your organization** 

* **This sample is provided for demonstration purposes. Make sure to conduct appropriate testing if derivating this code for your own use-cases. In general it is recommend to isolate development from production environments. Read more in the AWS Well Architected Framework https://aws.amazon.com/architecture/well-architected/**

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

