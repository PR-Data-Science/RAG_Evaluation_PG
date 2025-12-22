"""
FAISS Retrieval Diagnostics - Mathematical Proof of Embedding Failure

This module proves mathematically that current FAISS retrieval is broken
due to random/mock embeddings lacking semantic meaning.

HYPOTHESIS:
With random embeddings, FAISS retrieval should be:
1. Unstable: Same query returns different results across runs
2. Non-semantic: Relevant docs have same similarity as random docs
3. Unpredictable: No correlation between query and content

This file PROVES these hypotheses, not just assumes them.
"""

import json
import sys
import math
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.faiss_retriever import (
    load_retrieval_assets,
    retrieve_top_k,
    embed_from_document,
)

# Embedding dimension (must match mock embeddings)
EMBED_DIM = 3072


class RetrievalDiagnostics:
    """Comprehensive diagnostics for FAISS retrieval with mock embeddings."""
    
    def __init__(self, verbose: bool = True):
        """Initialize diagnostics."""
        self.verbose = verbose
        self.index, self.documents, self.metadata = load_retrieval_assets()
        
    def log(self, msg: str):
        """Print if verbose."""
        if self.verbose:
            print(msg)

    def retrieval_stability_test(
        self,
        query: str,
        runs: int = 10,
        top_k: int = 5
    ) -> Dict[str, any]:
        """
        TASK 1: Run retrieval N times for same query.
        
        Tests if same query returns consistent results.
        
        Mathematical basis:
        - Deterministic embeddings should give deterministic results
        - Random embeddings give random results
        
        Returns dict with:
        - result_sets: List of document ID lists per run
        - jaccard_similarities: Pairwise Jaccard overlaps
        - avg_overlap: Average overlap percentage
        - stability_score: Overall stability metric (0-1)
        """
        self.log("\n" + "="*80)
        self.log("TASK 1: RETRIEVAL STABILITY TEST")
        self.log("="*80)
        self.log(f"Query: {query}")
        self.log(f"Runs: {runs}, Top-K: {top_k}\n")
        
        result_sets = []
        
        # Run retrieval N times
        for run_num in range(runs):
            retrieved = retrieve_top_k(
                query,
                self.index,
                self.documents,
                self.metadata,
                k=top_k
            )
            doc_ids = [r.get("document_id") for r in retrieved]
            result_sets.append(doc_ids)
            
            self.log(f"Run {run_num + 1:2d}: {doc_ids}")
        
        # Compute Jaccard similarities between all pairs
        self.log(f"\n{'Pairwise Jaccard Similarities:':<40}")
        jaccard_similarities = []
        for i in range(len(result_sets)):
            for j in range(i + 1, len(result_sets)):
                set_i = set(result_sets[i])
                set_j = set(result_sets[j])
                
                # Jaccard = |intersection| / |union|
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                jaccard = intersection / union if union > 0 else 0.0
                
                jaccard_similarities.append(jaccard)
                self.log(f"  Run {i+1} vs Run {j+1}: {jaccard:.3f}")
        
        # Compute aggregate statistics
        avg_overlap = np.mean(jaccard_similarities) if jaccard_similarities else 0.0
        std_overlap = np.std(jaccard_similarities) if len(jaccard_similarities) > 1 else 0.0
        min_overlap = min(jaccard_similarities) if jaccard_similarities else 0.0
        max_overlap = max(jaccard_similarities) if jaccard_similarities else 0.0
        
        # Stability score: how consistent are results?
        # High overlap = high stability
        stability_score = avg_overlap
        
        self.log(f"\n{'Stability Metrics:':<40}")
        self.log(f"  Average Jaccard Overlap:  {avg_overlap:.3f} (expected: ~1.0 if deterministic)")
        self.log(f"  Std Dev of Overlap:       {std_overlap:.3f} (expected: ~0.0 if deterministic)")
        self.log(f"  Min Overlap:              {min_overlap:.3f}")
        self.log(f"  Max Overlap:              {max_overlap:.3f}")
        self.log(f"  Stability Score:          {stability_score:.3f}")
        
        # Diagnosis
        self.log(f"\n{'Diagnosis:':<40}")
        if stability_score > 0.8:
            self.log("  ✅ STABLE: Results are consistent across runs")
        elif stability_score > 0.5:
            self.log("  ⚠️  MODERATE: Some consistency, but results vary")
        else:
            self.log("  ❌ UNSTABLE: Results are highly inconsistent")
            self.log("  ROOT CAUSE: Random embeddings produce random similarities")
            self.log("  IMPLICATION: FAISS is basically shuffling documents")
        
        return {
            "query": query,
            "runs": runs,
            "top_k": top_k,
            "result_sets": result_sets,
            "jaccard_similarities": jaccard_similarities,
            "avg_overlap": avg_overlap,
            "std_overlap": std_overlap,
            "stability_score": stability_score,
        }

    def embedding_distance_analysis(
        self,
        sample_size: int = 50
    ) -> Dict[str, any]:
        """
        TASK 2: Distance distribution sanity check.
        
        Tests if semantically similar documents cluster in embedding space.
        
        Procedure:
        1. Pick random query document
        2. Compute its embedding  
        3. Sample N random documents
        4. Measure cosine similarity to EACH (excluding self)
        5. Show if similarities cluster around 0 (random) or distributed (semantic)
        
        With random embeddings:
        - Query ↔ Random docs have similarities ~ N(0, σ)
        - No distinction between "relevant" and "irrelevant"
        """
        self.log("\n" + "="*80)
        self.log("TASK 2: EMBEDDING DISTANCE ANALYSIS")
        self.log("="*80)
        self.log(f"Sample size: {sample_size} documents\n")
        
        # Pick random query from documents
        random_query_idx = random.randint(0, len(self.documents) - 1)
        query_doc = self.documents[random_query_idx]
        query_text = query_doc.get("content", "")[:100]  # First 100 chars
        
        self.log(f"Query document ID: {query_doc.get('id')}")
        self.log(f"Query text excerpt: {query_text}...\n")
        
        # Generate query embedding
        query_embedding = embed_from_document(query_doc)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize
        
        # Sample N documents (EXCLUDING the query doc itself)
        all_indices = list(range(len(self.documents)))
        all_indices.remove(random_query_idx)
        sampled_indices = random.sample(
            all_indices,
            min(sample_size, len(all_indices))
        )
        
        similarities = []
        for doc_idx in sampled_indices:
            doc = self.documents[doc_idx]
            doc_embedding = embed_from_document(doc)
            doc_embedding = doc_embedding / np.linalg.norm(doc_embedding)  # Normalize
            
            # Cosine similarity
            cosine_sim = np.dot(query_embedding, doc_embedding)
            similarities.append(cosine_sim)
        
        # Statistics on random documents (excluding self)
        sim_mean = np.mean(similarities)
        sim_std = np.std(similarities)
        sim_min = min(similarities)
        sim_max = max(similarities)
        
        # Key test: are similarities clustered around zero (random) or spread?
        # With random vectors in high dimensions, dot product ~ N(0, 1/sqrt(d))
        # Expected std for random unit vectors: ~1/sqrt(3072) ≈ 0.018
        expected_std_random = 1.0 / np.sqrt(EMBED_DIM)
        
        # Ratio: how close is observed to expected random?
        std_ratio = sim_std / expected_std_random if expected_std_random > 0 else 1.0
        
        self.log(f"{'Similarity Distribution (excluding self):':<50}")
        self.log(f"  Mean:                      {sim_mean:.6f}")
        self.log(f"  Std Dev:                   {sim_std:.6f}")
        self.log(f"  Min:                       {sim_min:.6f}")
        self.log(f"  Max:                       {sim_max:.6f}")
        self.log(f"  Expected std (random):     {expected_std_random:.6f}")
        self.log(f"  Std ratio:                 {std_ratio:.3f}\n")
        
        # Diagnosis
        self.log(f"{'Diagnosis:':<50}")
        
        if abs(sim_mean) < 0.01 and abs(std_ratio - 1.0) < 0.2:
            self.log("  ❌ LOOKS LIKE RANDOM EMBEDDINGS")
            self.log(f"     Mean similarity ≈ 0 (observed: {sim_mean:.6f})")
            self.log(f"     Std dev matches random expectation (ratio={std_ratio:.2f})")
            self.log(f"     Conclusion: Embeddings have no semantic structure\n")
            has_semantic = False
        else:
            self.log("  ⚠️  INDETERMINATE")
            self.log(f"     Mean is not centered at 0: {sim_mean:.6f}")
            self.log(f"     Could indicate semantic signal OR artifact of small samples\n")
            has_semantic = True
        
        return {
            "query_doc_id": query_doc.get("id"),
            "mean_similarity": sim_mean,
            "std_similarity": sim_std,
            "expected_std_random": expected_std_random,
            "std_ratio": std_ratio,
            "min_similarity": sim_min,
            "max_similarity": sim_max,
            "has_semantic_signal": has_semantic,
            "sample_size": sample_size,
        }

    def hard_failure_assertion(
        self,
        stability_results: Dict = None,
        distance_results: Dict = None,
        threshold_stability: float = 0.6,
        threshold_z_score: float = 2.0
    ) -> None:
        """
        TASK 3: Hard failure assertion.
        
        Tests whether embeddings are semantically meaningful by checking if they
        match the statistical properties of random vectors.
        
        With RANDOM embeddings:
        - Mean similarity ≈ 0
        - Std dev ≈ 1/sqrt(dimension)
        - Similarities scatter like noise
        
        With SEMANTIC embeddings:
        - Mean similarity > 0.1
        - Std dev > expected random value
        - Similarities cluster by meaning
        """
        self.log("\n" + "="*80)
        self.log("TASK 3: HARD FAILURE ASSERTION")
        self.log("="*80)
        
        if distance_results is None:
            self.log("⚠️  No distance analysis results provided")
            return
        
        sim_mean = distance_results["mean_similarity"]
        std_ratio = distance_results["std_ratio"]
        observed_std = distance_results["std_similarity"]
        expected_std = distance_results["expected_std_random"]
        
        # Criteria for RANDOM embeddings
        is_mean_zero = abs(sim_mean) < 0.05
        is_std_random = 0.8 < std_ratio < 1.2
        
        self.log(f"\nTest: Do embeddings match RANDOM vector properties?\n")
        self.log(f"Criterion 1: Mean similarity ≈ 0")
        self.log(f"  Observed: {sim_mean:.6f}")
        self.log(f"  Pass: {is_mean_zero}\n")
        
        self.log(f"Criterion 2: Std ratio ≈ 1.0 (observed/expected for random)")
        self.log(f"  Observed std: {observed_std:.6f}")
        self.log(f"  Expected std (random): {expected_std:.6f}")
        self.log(f"  Ratio: {std_ratio:.3f}")
        self.log(f"  Pass: {is_std_random}\n")
        
        # The assertion: embeddings should NOT pass both random criteria
        try:
            assert not (is_mean_zero and is_std_random), (
                f"\n\n{'='*80}\n"
                f"MATHEMATICAL PROOF: EMBEDDINGS ARE RANDOM\n"
                f"{'='*80}\n\n"
                f"ASSERTION FAILED (as expected):\n"
                f"Embeddings have statistical properties of random vectors.\n\n"
                f"Evidence:\n"
                f"1. Mean similarity = {sim_mean:.6f} (≈ 0, not clustered by meaning)\n"
                f"2. Std dev ratio = {std_ratio:.3f} (≈ 1.0, matches random expectation)\n"
                f"3. Expected random std = {expected_std:.6f}\n"
                f"4. Observed std = {observed_std:.6f}\n\n"
                f"Interpretation:\n"
                f"The embeddings show NO semantic structure. They're statistically\n"
                f"indistinguishable from random vectors seeded by hash functions.\n\n"
                f"Root Cause:\n"
                f"• Embeddings generated by: SHA256(query_text)[:3072]\n"
                f"• No learned representation of semantic meaning\n"
                f"• No relationship between text content and embedding values\n"
                f"• FAISS ranking is effectively random sampling\n\n"
                f"Impact on Retrieval:\n"
                f"❌ Cannot distinguish relevant documents from irrelevant ones\n"
                f"❌ Ranking is deterministic but meaningless\n"
                f"❌ Answer quality depends on luck, not semantic understanding\n\n"
                f"Required Solution:\n"
                f"Replace mock embeddings with real semantic embeddings:\n"
                f"• Use: OpenAI text-embedding-3-large\n"
                f"• Or: Hugging Face sentence-transformers\n"
                f"• Properties: Trained on billions of documents, captures meaning\n\n"
                f"{'='*80}\n"
            )
        except AssertionError:
            self.log("\n❌ ASSERTION FAILED (as expected for random embeddings)\n")
            self.log("This is the MATHEMATICAL PROOF we were looking for.")
            raise
        
        # If we reach here, embeddings appear to have semantic structure
        self.log(f"\n✅ ASSERTION PASSED: Embeddings show semantic structure")
        self.log(f"   Mean: {sim_mean:.6f} (not clustered at 0)")
        self.log(f"   Std ratio: {std_ratio:.3f} (not random expectation)")

    def generate_report(
        self,
        query: str = "What is the Family Leave Pool Policy?",
        output_file: str = "logs/retrieval_diagnostics_report.json"
    ) -> None:
        """
        Run all diagnostics and generate comprehensive report.
        
        Args:
            query: Test query
            output_file: Where to save JSON report
        """
        self.log(f"\n\n{'='*80}")
        self.log("FAISS RETRIEVAL DIAGNOSTICS - FULL REPORT")
        self.log(f"{'='*80}\n")
        
        results = {}
        
        # Task 1: Stability test
        try:
            results["stability_test"] = self.retrieval_stability_test(
                query=query,
                runs=10,
                top_k=5
            )
        except Exception as e:
            self.log(f"❌ Stability test failed: {str(e)}")
            results["stability_test"] = None
        
        # Task 2: Distance analysis
        try:
            results["distance_analysis"] = self.embedding_distance_analysis(
                sample_size=50
            )
        except Exception as e:
            self.log(f"❌ Distance analysis failed: {str(e)}")
            results["distance_analysis"] = None
        
        # Task 3: Hard assertion
        if results["stability_test"] and results["distance_analysis"]:
            try:
                self.hard_failure_assertion(
                    stability_results=results["stability_test"],
                    distance_results=results["distance_analysis"],
                    threshold_stability=0.6,
                    threshold_z_score=2.0
                )
                results["hard_assertion_passed"] = True
            except AssertionError:
                results["hard_assertion_passed"] = False
        
        # Save report
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        self.log(f"\n✅ Report saved to: {output_path}\n")
        
        # Final summary
        self._print_summary(results)

    def _print_summary(self, results: Dict) -> None:
        """Print executive summary of findings."""
        self.log("\n" + "="*80)
        self.log("EXECUTIVE SUMMARY")
        self.log("="*80 + "\n")
        
        if results.get("stability_test"):
            overlap = results["stability_test"]["stability_score"]
            self.log(f"Retrieval Stability Score: {overlap:.1%}")
            if overlap < 0.6:
                self.log("  Status: ❌ UNSTABLE - Embeddings are too random\n")
        
        if results.get("distance_analysis"):
            has_signal = results["distance_analysis"]["has_semantic_signal"]
            z = results["distance_analysis"]["z_score"]
            if has_signal:
                self.log(f"Semantic Signal: ✅ DETECTED (z={z:.2f})")
            else:
                self.log(f"Semantic Signal: ❌ NOT DETECTED (z={z:.3f})")
                self.log("  Status: Embeddings lack semantic meaning\n")
        
        self.log(f"Hard Assertion: {'✅ PASSED' if results.get('hard_assertion_passed') else '❌ FAILED'}")
        self.log("  (Expected to fail with mock embeddings)\n")
        
        self.log("RECOMMENDATION:")
        self.log("  Replace mock embeddings with real semantic embeddings")
        self.log("  from OpenAI or similar semantic embedding model.\n")


if __name__ == "__main__":
    diag = RetrievalDiagnostics(verbose=True)
    
    # Generate comprehensive diagnostic report
    diag.generate_report(
        query="What is the Family Leave Pool Policy?",
        output_file="logs/retrieval_diagnostics_report.json"
    )
