from sentence_transformers import SentenceTransformer, util
import torch
import re

class TextClassifier:
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        # Load multilingual embedding model (supports Burmese + English)
        self.model = SentenceTransformer(model_name)
        
        # Define categories
        self.categories = [
            'Social',
            'Entertainment',
            'Product',
            'Service',
            'Business',
            'Sports',
            'Science',
            'Technology',
            'Education',
            'Culture',
            'History',
            'Health',
            'Environmental'
        ]
        
        # Pre-compute category embeddings
        self.category_embeddings = self.model.encode(self.categories, convert_to_tensor=True)

    def extract_hashtags(self, text: str):
        """Extract all hashtags (without # sign)."""
        return [tag.strip("#").lower() for tag in re.findall(r"#\w+", text)]

    def classify(self, text: str):
        """Extract hashtags, classify each, and return results."""
        hashtags = self.extract_hashtags(text)
        results = []
        for tag in hashtags:
            text_emb = self.model.encode(tag, convert_to_tensor=True)
            sims = util.cos_sim(text_emb, self.category_embeddings)[0]
            best_idx = torch.argmax(sims).item()
            results.append({
                "hashtag": f"#{tag}",
                "category": self.categories[best_idx],
                "score": round(sims[best_idx].item(), 3)
            })
        return results


if __name__ == "__main__":
    classifier = TextClassifier()

    # Try any example text
    text = "Happy #independant_day everyone! Excited for the #football and #AI revolution!"
    
    results = classifier.classify(text)
    print("\nInput text:", text)
    print("\nClassification results:")
    for r in results:
        print(f"  {r['hashtag']} → {r['category']} (score={r['score']})")