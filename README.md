# 🧠 EmpaCare: A Multimodal Empathetic Mental Health Chatbot

EmpaCare is a GenAI-powered mental health chatbot that uses multimodal signals (text and audio) to detect user emotion and depression severity (PHQ-8), and responds with contextually supportive and empathetic dialogue. Built on top of DAIC-WOZ dataset, the system combines LLaMA-3 + LoRA for response generation, LangChain + FAISS for RAG, and BERT + OpenSMILE for emotion detection.

---

## 🧪 Key Features

- 🎙️ Multimodal Emotion & Depression Detection (Text + Audio)
- 🤗 Empathetic Response Generation using LLaMA-3 fine-tuned with LoRA
- 🧠 Context-Aware Dialogue via LangChain + FAISS (RAG)
- 🩺 Personalized support aligned with CBT principles
- 🌐 Gradio-based Web UI for demo and deployment

---

## 📦 Dataset

**DAIC-WOZ**  
- Mental health interview corpus (text transcripts, audio features, visual features, PHQ-8 scores)  
- Download: [https://dcapswoz.ict.usc.edu/](https://dcapswoz.ict.usc.edu/)

---

## 🔧 Tech Stack

| Component | Tools Used |
|----------|-------------|
| Language Model | LLaMA-3 + LoRA |
| Emotion Detection | BERT, OpenSMILE |
| Dialogue Manager | LangChain, FAISS |
| UI | Gradio |
| Frameworks | PyTorch, HuggingFace Transformers |
| Evaluation | RMSE, F1, Empathy Score, Human Eval |

---

## 🧱 System Architecture


+-----------------------------+
|    User Input (Text/Audio) |
+-----------------------------+
             ↓
+-----------------------------+
|   Multimodal Emotion Model  | ← BERT + OpenSMILE
+-----------------------------+
             ↓
+-----------------------------+
|  Emotion/PHQ Classifier     | ← MLP
+-----------------------------+
             ↓
+-----------------------------+
| Dialogue Manager (RAG)      | ← LangChain + FAISS
+-----------------------------+
             ↓
+-----------------------------+
| LLaMA-3 + LoRA (Empathetic) |
+-----------------------------+
             ↓
|     Contextual Response     |
+-----------------------------+


⸻

🚀 Getting Started

1. Clone the Repository

git clone https://github.com/your-username/EmpaCare.git
cd EmpaCare

2. Install Dependencies

pip install -r requirements.txt

3. Prepare DAIC-WOZ Data

Download the dataset and organize as follows:

data/
├── daic_woz_transcripts/
├── audio/
└── phq_scores.csv

4. Extract Features

python preprocessing/text_preprocess.py
python preprocessing/audio_feature_extractor.py

5. Train Emotion & PHQ Classifier

python models/train_classifier.py

6. Fine-Tune LLaMA-3 with LoRA

python models/lora_finetune.py --base_model llama-3 --dataset EmpatheticDialogues

7. Launch the Chatbot

python app/gradio_ui.py


⸻

📊 Evaluation
	•	PHQ-8 Score Prediction: RMSE, MAE
	•	Emotion Detection: F1-Score (macro/weighted)
	•	Response Quality: BLEU, ROUGE
	•	Empathy: Comparison with EmpatheticDialogues + Human Ratings

⸻

📚 Key References
	1.	Majumder et al. (2019) – Multimodal Transformer for Emotion Recognition
	2.	Rashkin et al. (2019) – EmpatheticDialogues Dataset
	3.	Liu et al. (2023) – Mixtral and MixMoE LLMs
	4.	Althoff et al. (2016) – Counseling Conversations for Mental Health
	5.	Stas et al. (2023) – Parameter-Efficient LoRA Fine-Tuning

⸻

🧑‍💻 Contributing

We welcome community contributions! If you’d like to:
	•	Improve emotion modeling
	•	Add visual modality
	•	Tune response generation
	•	Enhance evaluation

Please open an issue or submit a pull request.

⸻

📄 License

MIT License — feel free to use, modify, and share!

⸻

💬 Acknowledgements

Thanks to the USC DAIC-WOZ team, HuggingFace community, LangChain developers, and researchers in affective computing for making this work possible.
----
