# QD-LLM: Quantum Knowledge Distillation for Large Language Models

QD-LLM is a novel framework that distills the power of large language models (LLMs) into compact quantum student models using variational quantum circuits. This enables efficient deployment of NLP models on resource-constrained hardware while maintaining high performance.



📄 Paper: Quantum Knowledge Distillation for Large Language Models

🧑‍💻 Authors: Lingxiao Li*, Yihao Wang*, Jiacheng Fan, Jing Li, Sujuan Qin, Qiaoyan Wen, Fei Gao

🏫 Affiliations: Beijing University of Posts and Telecommunications


🚀 Highlights:

🔍 Distills LLMs like LLaMA2/3, OPT, BLOOMZ into quantum circuits

🌌 Achieves comparable or better performance than TinyBERT, DistilBERT, MiniLLM, etc.

⚡ Only 9K parameters in the quantum model (~0.02% of classical distillation baselines)

💻 Classical simulation + 🧪 Deployment on real quantum hardware (Quafu superconducting processor)

🧠 Quantum-Inspired Classical Algorithm: even classical simulations show strong compression + accuracy 


🧪 Benchmark Tasks
QD-LLM is evaluated on:
| Task              | Dataset Size | Classes | Avg. Text Length |
| ----------------- | ------------ | ------- | ---------------- |
| Emotion Analysis  | 24,000       | 2       | 33.17            |
| Hiding Detection  | 10,000       | 2       | 10.80            |
| Thematic Analysis | 20,000       | 4       | 13.96            |


These datasets can be seen in the Dataset file.

🌐 Open-Source LLMs Used

| Model    | Links                                                             |
| ---------| ----------------------------------------------------------------- |
| LLaMA 2  | Meta AI (https://github.com/meta-llama/llama)                     |
| LLaMA 3  | Meta AI (https://ai.meta.com/llama/)                              |
| OPT      | OPT GitHub (https://github.com/facebookresearch/metaseq)          |
| BLOOMZ   | BLOOMZ on HuggingFace (https://huggingface.co/bigscience/bloomz)  |



🧊 Real-device 
The details of the deployment are shown in real_device_baihua. We thank the Quafu Quantum Cloud Platform and the Beijing Institute of Technology for their support during the real-device evaluation.

📬 Contact
For questions or collaborations:

Lingxiao Li: lingxiao_li@bupt.edu.cn




