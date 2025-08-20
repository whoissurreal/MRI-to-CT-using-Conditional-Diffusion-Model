🧠 MRI-to-CT Synthesis with Conditional Diffusion Models

📌 Overview

This project implements a conditional diffusion-based framework for cross-modality medical image synthesis, specifically converting MRI scans into CT images. The goal is to provide clinicians with complementary diagnostic information without subjecting patients to additional radiation exposure. By leveraging generative diffusion models, advanced feature extractors, and attention mechanisms, the framework generates CT images that are both structurally accurate and perceptually realistic.

⸻

🔑 Key Features
	•	Hybrid Architecture: Residual U-Net backbone enhanced with DINOv2 feature extractors and CBAM attention modules.
	•	Conditional Diffusion Process: Gradual refinement of MRI inputs into CT outputs while preserving anatomical details.
	•	Loss Function: Combination of L1 reconstruction loss and perceptual loss to balance pixel accuracy and visual quality.
	•	Reusable Components: Modular implementation with scripts for dataset handling, training, losses, noise scheduling, and attention.

🚀 Usage
Install dependencies:

	1.	pip install -r requirements.txt

pip install -r requirements.txt

Train the model:

	2.	python train.py --config config.json


Run inference:

	3.	Run inference:python run.py --input MRI_image.nii.gz --output CT_output.nii.gz





