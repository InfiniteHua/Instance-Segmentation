# AI-Based Flower Instance Detection and Picking Optimization
## Chapter 1 - Task Introduction
### 1.PROBLEM STATEMEN
When flowers are harvested in the greenhouse or from the field, they are typically collected into a boundled heap. Before they can be processed by a machine, the flowers must currently be picked up manually, one by one, from the pile and placed into the machine. This task is labor-intensive and complicated further by the fact that flowers are often entangled.

Thus, automating this process would significantly reduce manual labor. The critical point is that the flower must first be individually identified. This assignment focuses on designing a setup to capture and map the flowers, using AI to detect the individual flower instances, and determining which flower is best suited to be picked first from the pile (output the pickup coordinates for the manipulator).
### 2.Solution
This solution is divided into three methods: 
* **Preprocessing**:  `Roboflow` is an online website helps with various data processing such as Annotation and Data Augmentation.
* **Instance Segmentation**: A `Mask-RCNN` neural network is trained to detect all stem segments.
* **Post Processing**: (1) The First Post-Processing algorithm `Stem Pairing` is designed to match all stems from each flower. (2) The Second Post-Processing algorithm `Top Selection` is designed to match all stems from each flower.
## Chapter 2 - Preprocessing
## Chapter 3 - Instance Segmentation
## Chapter 4 - Post Processing