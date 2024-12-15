# Cybersecurity
This repository hosts the project developed for the Cybersecurity course, part of the Master's program in Artificial Intelligence at the University of Bologna, during the 2024-2025 academic year.

The project is designed to meet the requirements for the final exam by addressing the following task:

Use sparsity techniques to detect if a dataset has been poisoned.

_Hypothesis_

Poisoned samples are resilient to misclassification errors. By introducing noise in the
network, it should be possible to find the adversarial samples.

Goal:
• Find or build a poisoned dataset of malware (for example using https://github.com/ClonedOne/MalwareBackdoors)
• Train a neural network as a malware detector
• Add noise to the internal weight of the network (or sparsify the network)
• Check for a correlation between the classification result after the added noise and the poisoned samples

References
https://www.usenix.org/system/files/sec21-severi.pdf
https://arxiv.org/abs/1803.03635

