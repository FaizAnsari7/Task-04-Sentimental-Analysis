
"""
Task-04: Sentiment Analysis of Social Media Data
Goal: Analyze and visualize sentiment patterns in social media posts.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- 1. Load data ----------
csv_path = "social_media.csv"

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    print(f"'{csv_path}' not found — generating a synthetic sample of posts.")

    sample_posts = [
        "I absolutely love the new update, it's fantastic!",
        "This app keeps crashing, so frustrating and disappointing.",
        "Customer service was okay, nothing special.",
        "Best purchase I've made this year, highly recommend!",
        "Terrible experience, I want a refund immediately.",
        "It's fine I guess, does the job.",
        "Amazing quality and super fast delivery, thank you!",
        "Worst product ever, complete waste of money.",
        "Not bad, could be better but I'm satisfied overall.",
        "Horrible support team, nobody replied for a week.",
        "The new design looks great and works smoothly.",
        "Meh, I expected more for the price.",
        "So happy with this brand, will buy again!",
        "Really disappointed, it broke after two days.",
        "Pretty average, no strong feelings either way."
    ]

    rng = np.random.default_rng(3)
    topics = rng.choice(["BrandX", "BrandY"], size=len(sample_posts))

    df = pd.DataFrame({
        "text": sample_posts,
        "topic": topics
    })

print(df.head())

# ---------- 2. Sentiment Scoring ----------
POSITIVE_WORDS = {
    "love", "great", "fantastic", "amazing", "best",
    "happy", "good", "excellent", "recommend", "smooth",
    "fast", "thank", "satisfied", "nice", "awesome"
}

NEGATIVE_WORDS = {
    "hate", "terrible", "worst", "bad", "frustrating",
    "disappointing", "disappointed", "horrible", "waste",
    "broke", "crashing", "refund", "poor", "slow", "awful"
}

def sentiment_score(text: str) -> int:
    """Return +1 (positive), -1 (negative), or 0 (neutral)."""
    words = re.findall(r"[a-zA-Z']+", text.lower())

    pos = sum(w in POSITIVE_WORDS for w in words)
    neg = sum(w in NEGATIVE_WORDS for w in words)

    if pos > neg:
        return 1
    elif neg > pos:
        return -1
    return 0


def sentiment_label(score: int) -> str:
    return {1: "Positive", 0: "Neutral", -1: "Negative"}[score]


df["sentiment_score"] = df["text"].apply(sentiment_score)
df["sentiment"] = df["sentiment_score"].apply(sentiment_label)

print("\n", df[["text", "sentiment"]])

# ---------- 3. Visualize Sentiment Distribution ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sentiment_counts = (
    df["sentiment"]
    .value_counts()
    .reindex(["Positive", "Neutral", "Negative"], fill_value=0)
)

colors = {
    "Positive": "#55A868",
    "Neutral": "#8C8C8C",
    "Negative": "#C44E52"
}

# Overall sentiment distribution
axes[0].bar(
    sentiment_counts.index,
    sentiment_counts.values,
    color=[colors[s] for s in sentiment_counts.index]
)

axes[0].set_title("Overall Sentiment Distribution")
axes[0].set_ylabel("Number of Posts")

# Sentiment by topic/brand
if "topic" in df.columns:
    pivot = (
        df.groupby(["topic", "sentiment"])
        .size()
        .unstack(fill_value=0)
    )

    pivot = pivot.reindex(
        columns=["Positive", "Neutral", "Negative"],
        fill_value=0
    )

    pivot.plot(
        kind="bar",
        stacked=True,
        ax=axes[1],
        color=[colors[c] for c in pivot.columns]
    )

    axes[1].set_title("Sentiment by Topic/Brand")
    axes[1].set_ylabel("Number of Posts")
else:
    axes[1].axis("off")

plt.tight_layout()

# ---------- 4. Save output ----------
out_path = "task4_sentiment.png"
plt.savefig(out_path, dpi=150)
print(f"\nSaved chart to {out_path}")

print("\nSentiment counts:\n", sentiment_counts)

plt.show()
