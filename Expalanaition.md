Environment Setup (Cells 1–3)
The pipeline begins by confirming that a GPU is available and visible to the system using Python's subprocess module to call nvidia-smi. If no GPU is detected, an environmental error is raised immediately to prevent the model from silently falling back to the CPU, which would be too slow for a network of this size.

Once the hardware is verified, the ultralytics package is imported to provide the core YOLOv8 architecture, training loops, and validation utilities. The system runs an automated compatibility check to ensure PyTorch, CUDA, and the library are working together seamlessly. Finally, all required data science, plotting, and file-system modules are imported at the top level to prevent any missing-dependency crashes later in execution.

Dataset Mounting and Path Configuration (Cell 4)
Rather than downloading the image dataset dynamically at runtime, this step interfaces directly with Kaggle’s native "Add Data" storage structure. When a dataset is attached, the platform mounts it as a read-only directory under the root input path. The code defines the explicit absolute path to the waste-sorting dataset and immediately inspects the directory contents. If the folder cannot be found (often due to an incorrect URL slug or unmounted data), a file exception is triggered early to stop execution before the pipeline fails deeper in the training loop.

Dataset Structure Verification (Cell 5)
YOLOv8 enforces a strict directory layout to properly map images to their respective bounding box labels. The root directory must contain a configuration file (data.yaml), alongside structured subfolders for training and validation data splits. Within these splits, images and text labels must be separated into paired subdirectories, where each plain text file contains object indices and normalized coordinates matching an individual image. This cell explicitly checks for the existence of the configuration file and reads its properties out loud to guarantee the layout matches what the model architecture expects.

Fixing Absolute Paths in data.yaml (Cell 6)
Datasets prepared locally often use relative directory paths inside their configuration file, which causes failures in cloud environments like Kaggle because the working directory is separated from the read-only input mount point. This cell resolves this issue by programmatically loading the configuration data, appending the absolute read-only input paths to the training and validation keys, and validating whether an optional test split directory is physically present. If a test split is missing, its key is safely stripped out to prevent tracking errors. The corrected configuration is then compiled and written out as a brand-new file into the cloud environment's writable directory.

Loading the Pretrained Model (Cell 7)
To maximize training efficiency and feature accuracy, the pipeline leverages transfer learning rather than initializing the network with random weights. The small variant checkpoint of the network (yolov8s.pt) is loaded into memory, automatically downloading its baseline weights if they are not cached. Because this model was originally trained on the extensive COCO dataset, it already possesses an advanced understanding of low-level visual features like edges, shapes, and complex textures. Fine-tuning from this baseline dramatically reduces the volume of training data and epochs required to achieve high precision.

Training (Cell 8)
This section executes the core training loop for the object detection model. It clears out old tracking artifacts to ensure clean metrics and reloads the fresh checkpoint so the block can be executed repeatedly without carrying over residual state. The training function is configured with a set of parameters tailored for cloud environments:

It forces a standardized input resolution of 640 pixels.

It leverages an automated batch size selector to safely maximize GPU memory utilization up to a stable limit.

Early stopping is introduced to automatically halt the training process if validation accuracy stalls for a consecutive number of cycles, preventing wasteful computation.

Worker threads are restricted to avoid system kernel crashes, and built-in image augmentation techniques—such as mosaic stitching, spatial flips, and color alterations—are activated to artificially diversify the training data.

Confirming Results and Validation (Cells 9–10)
Following the completion of the training cycles, the script checks the local filesystem to verify that the weights and training log files were successfully saved. It extracts the path of the top-performing checkpoint weight file and initializes a dedicated validation instance of the model. The system determines whether to evaluate against the dedicated test split or fall back to the standard validation data. It then extracts key performance metrics, specifically recording mean Average Precision at standard intersection thresholds, alongside pure precision and recall scores to evaluate overall localization and classification accuracy.

Inference on Test Images (Cell 11)
To visually confirm real-world performance, the pipeline targets the directory holding unseen test images. The model processes these samples using a baseline confidence threshold and an intersection-over-union suppression filter to drop duplicate bounding box predictions. If test images are present, the network generates and saves annotated export copies of the graphics, complete with mapped bounding boxes and prediction probability scores, allowing for direct qualitative evaluation of the model's accuracy.

Packaging Weights for Download (Cell 12)
Because cloud execution environments wipe temporary data storage when a session ends, the best-performing and final model weight files must be archived for external use. This cell creates a compressed archive folder appended with a precise date and time stamp. It packages the compiled weight files along with the historical metrics data sheet into a single downloadable zip file, delivering a single clean package containing all execution artifacts.

Model Export (Cell 13)
For real-world deployment on target hardware, the standard training format is converted into a highly optimized runtime engine. The script targets an export to the lightweight NCNN framework, which is tailored specifically for rapid inference on embedded devices and mobile processors without external software dependencies. The cell handles system-level compilation installations, manages required Python conversion packages, and validates all module imports. If any native system component fails to install, the pipeline falls back to an ONNX structure to ensure a functional model format is successfully exported regardless of environmental limitations.

Training Dashboard (Cell 14)
This cell reads the metric data sheet compiled automatically during execution and builds a multi-panel visualization using plotting libraries. It features an intelligent string-matching helper to find the correct data columns, protecting the pipeline against format shifts across different software versions. The script calculates peak metrics from the best-performing epoch and outputs a comprehensive diagnostic dashboard, saving a clean visual summary tracking classification, bounding box adjustment, and structural loss behaviors over the complete training history.

Confusion Matrix (Cell 14b)
The final step re-runs validation checks with automated plotting flags turned on to map the system's explicit confusion matrix. This matrix breaks down predictions at a class-specific level, contrasting ground-truth annotations directly against predicted boundaries. The resulting graphic is rendered inline inside the system workspace, providing a clear visual representation that shows exactly which classes the model detects perfectly and which specific labels are prone to being misclassified.
