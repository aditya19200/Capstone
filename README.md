🧠 XAI-Enabled Knowledge Graph-Driven Annotation Framework

A next-generation intelligent data annotation platform that integrates Explainable AI (XAI), Knowledge Graphs, and Active Learning to improve the quality, transparency, and efficiency of labeled datasets—especially for clinical and healthcare text data.

High-quality labeled datasets are critical for building robust machine learning models, particularly in sensitive domains such as healthcare. Traditional annotation tools focus mainly on labeling speed and workflow but lack:
	•	Explainability of AI predictions
	•	Ontology-driven label consistency
	•	Active learning for efficient annotation
	•	Collaborative consensus mechanisms

This project addresses these gaps by proposing an XAI-Enabled Knowledge Graph-Driven Annotation Framework that assists human annotators using explainable model predictions and structured domain knowledge.

🏟️ System Architecture (High Level)
Frontend (React + Tailwind)
        ↓
FastAPI Backend (REST APIs)
        ↓
ML Layer (ClinicalBERT + XAI)
        ↓
Knowledge Graph (Neo4j)
        ↓
Storage & Auth (Supabase)

🧩 Core Components

1️⃣ Frontend (Annotation Dashboard)
	•	Framework: React + Vite
	•	Styling: Tailwind CSS
	•	Features:
	•	Clinical text annotation workspace
	•	AI-predicted labels with confidence scores
	•	Token-level explanations (XAI)
	•	Knowledge Graph explorer
	•	Human actions: Accept / Modify / Flag Uncertain

2️⃣ Machine Learning Layer
	•	Model: emilyalsentzer/Bio_ClinicalBERT
	•	Why this model?
	•	Pretrained on real clinical notes (MIMIC-III)
	•	Strong performance on medical NLP tasks
	•	Supports explainability and active learning
	•	Tasks:
	•	Clinical text classification
	•	Confidence estimation
	•	Uncertainty sampling for active learning

  3️⃣ Explainable AI (XAI)
	•	Token-level importance highlighting
	•	Helps annotators understand why a label was predicted
	•	Improves trust and reduces blind acceptance of AI outputs

  4️⃣ Active Learning Engine
	•	Model labels all samples automatically
	•	Only low-confidence / high-uncertainty samples are sent to human reviewers
	•	Human feedback is used to:
	•	Update labels
	•	Improve future predictions
	•	Reduces annotation workload significantly

  5️⃣ Knowledge Graph (Neo4j)
	•	Stores medical ontology (e.g., Disease → Cardiovascular → Myocardial Infarction)
	•	Ensures hierarchical and semantic label consistency
	•	Enables graph-based reasoning and exploration

  6️⃣ Backend API
	•	Framework: FastAPI (Python)
	•	Responsibilities:
	•	ML inference endpoints
	•	Active learning orchestration
	•	Knowledge graph queries
	•	Annotation data management

  7️⃣ Authentication & Storage
	•	Auth: Supabase Authentication
	•	Storage:
	•	User metadata
	•	Annotation results
	•	Dataset versions
	•	Keeps ML pipeline independent from auth logic

  8️⃣ Containerization
	•	Docker used for:
	•	Reproducible environment
	•	Easy setup for teammates
	•	Consistent ML + backend execution
