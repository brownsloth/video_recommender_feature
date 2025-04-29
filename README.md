# Short Video Recommender System (Prototype)

This project simulates a personalized short video recommendation system using real video transcripts (YouTube Shorts) and article reading behavior (AG News).

## Features
- Downloads short YouTube videos from a channel using `yt-dlp`.
- Transcribes videos using OpenAI's Whisper model.
- Uses AG News as article dataset.
- Simulates user reading history and preferences.
- Recommends videos to users using TF-IDF embeddings + cosine similarity.
- Outputs top-K personalized recommendations for each user.

## Setup

```bash
pip install -r requirements.txt
brew install ffmpeg  # or apt install ffmpeg
```

## Run pipeline

```
python download_videos.py
python transcribe_videos.py
python main.py
```

## Directory structure

```
.
├── download_videos.py         # Download short videos from YouTube channel
├── transcribe_videos.py       # Transcribe all .mp4 videos using Whisper
├── recommend_videos.py        # Recommend top-K videos per user
├── video_transcripts.json     # Output of transcription
├── videos/                    # Downloaded short videos
```

## Sample output

```
Recommendations for user_123:
1. video_a.mp4 (0.622)
2. video_b.mp4 (0.589)
```


---

## 🚀 3. Roadmap: What We Can Do Next (Next-Level System)

Here’s a breakdown of how to evolve from prototype → production:

---

### 📈 **A. ML Modeling Improvements**
| Feature | Description |
|--------|-------------|
| ✅ TF-IDF | Good baseline |
| 🔜 Sentence-BERT | Better semantic similarity |
| 🔜 GenAI Embeddings | Use `text-embedding-ada-002`, Mistral, or `BGE` |
| 🔜 Two-Tower Model | User tower + video tower (deep, flexible) |
| 🔜 Hierarchical Tower | Session → User + Item → Video Contexts |
| 🔜 Contrastive Learning | User–positive–negative triplets |
| 🔜 Session Modeling | Include recent views, dwell time, categories |

---

### ☁️ **B. Full Production Pipeline on AWS**

| Component | Tool/Service |
|----------|---------------|
| **Data Ingestion** | S3 + Lambda |
| **Data Validation** | Great Expectations / Deequ |
| **Feature Store** | AWS SageMaker Feature Store |
| **Training Pipeline** | SageMaker Pipelines |
| **Model Registry** | SageMaker Model Registry |
| **Model Serving** | SageMaker Endpoint / ECS |
| **Monitoring** | CloudWatch + custom metrics |
| **Drift Detection** | Monitor feature distribution, KL-divergence |
| **Retraining Trigger** | Lambda on S3 upload / schedule |
| **Security** | IAM Roles, S3 bucket policies, encryption, VPC config |

---

### 🧠 **C. Add GenAI + LLM-based Enhancements**
- Use LLM to:
  - Summarize videos.
  - Enrich user/article metadata.
  - Rewrite cold-start video titles for better matching.
- Use `embedding + reranking` with LLM (Hybrid Search).
- Session-aware prompts.

---

### 🌐 **D. Build a Minimal Web App (User Simulation)**

Use **Gradio** or **Streamlit**:

```python
import gradio as gr

def recommend_for_user(user_id):
    profile = user_profiles.get(user_id)
    if not profile:
        return "User not found"
    idxs, sims = recommend_videos(profile, video_embeddings)
    results = [
        f"{video_filenames[i]} (score: {s:.3f})\n{video_texts[i][:80]}..."
        for i, s in zip(idxs, sims)
    ]
    return "\n\n".join(results)

gr.Interface(fn=recommend_for_user, inputs="text", outputs="text").launch()
```