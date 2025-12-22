"""OpenAI Embeddings Client

This module provides a simple, no-dependency-on-LangChain client for OpenAI embeddings.
It handles API key validation, text trimming, error handling, and cost tracking.

Cost Notes:
- text-embedding-3-large: $0.13 per 1M tokens (as of 2025)
- Input text is trimmed to ~8000 tokens per request to stay safe
- Batching multiple texts into one API call saves costs (not yet implemented)

Why we trim text:
- OpenAI embeddings have a max context of ~8000 tokens
- Trimming prevents API errors and ensures consistency
"""

import os
import logging
from typing import List
import numpy as np

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("openai package required: pip install openai")
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # If dotenv is not installed, env vars may still be provided by the environment
    pass

LOGGER = logging.getLogger(__name__)

# Constants
EMBED_DIM = 3072  # text-embedding-3-large always returns 3072 dimensions
MODEL_NAME = "text-embedding-3-large"
MAX_TOKENS_PER_CHUNK = 8000  # Safe upper bound for input text
APPROX_TOKENS_PER_CHAR = 0.25  # Rough estimate: 1 token ≈ 4 characters


class OpenAIEmbeddingClient:
    """Client for generating embeddings using OpenAI's API.
    
    Usage:
        client = OpenAIEmbeddingClient()
        embedding = client.embed_text("Hello world")  # Returns list[float] of length 3072
    
    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set
        ValueError: If input text is empty
        RuntimeError: If API call fails
    """
    
    def __init__(self, model: str = MODEL_NAME, api_key: str = None):
        """Initialize the OpenAI embedding client.
        
        Args:
            model: OpenAI model to use (default: text-embedding-3-large)
            api_key: Optional API key (uses OPENAI_API_KEY env var if not provided)
            
        Raises:
            EnvironmentError: If OPENAI_API_KEY environment variable is not set
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            LOGGER.error("OPENAI_API_KEY environment variable not set")
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is required. "
                "Set it with: export OPENAI_API_KEY='your-key-here'"
            )
        
        self.model = model
        self.client = OpenAI(api_key=api_key)
        LOGGER.info(f"OpenAIEmbeddingClient initialized with model: {model}")
    
    def _trim_text(self, text: str, max_tokens: int = MAX_TOKENS_PER_CHUNK) -> str:
        """Trim text to safe token limit to avoid API errors.
        
        Args:
            text: Text to trim
            max_tokens: Maximum tokens allowed
            
        Returns:
            Trimmed text (same if under limit)
            
        Note:
            This uses character count as a rough proxy for token count.
            For precise token counting, use tiktoken library (not imported to avoid dependencies).
        """
        max_chars = int(max_tokens / APPROX_TOKENS_PER_CHAR)
        if len(text) > max_chars:
            LOGGER.warning(
                f"Text trimmed from {len(text)} to {max_chars} characters "
                f"(approx {max_tokens} tokens). Some content was discarded."
            )
            return text[:max_chars]
        return text
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list[float] of length 3072
            
        Raises:
            ValueError: If text is empty or None
            RuntimeError: If API call fails
        """
        # Validate input
        if not text or not isinstance(text, str):
            LOGGER.error(f"Invalid input: text must be non-empty string, got {type(text)}")
            raise ValueError(
                f"Text must be a non-empty string, got {type(text).__name__}: {repr(text)[:100]}"
            )
        
        # Trim text safely
        trimmed = self._trim_text(text.strip())
        
        if not trimmed:
            LOGGER.error("Text is empty after trimming")
            raise ValueError("Text is empty or only whitespace")
        
        try:
            LOGGER.debug(f"Calling OpenAI API for embedding (text length: {len(trimmed)})")
            response = self.client.embeddings.create(
                model=self.model,
                input=trimmed,
            )
            
            # Extract embedding from response
            if not response.data or len(response.data) == 0:
                LOGGER.error("No embedding returned from API")
                raise RuntimeError("OpenAI API returned empty response")
            
            embedding = response.data[0].embedding
            
            # Validate output
            if not isinstance(embedding, list) or len(embedding) != EMBED_DIM:
                LOGGER.error(
                    f"Invalid embedding shape: expected list of {EMBED_DIM}, "
                    f"got {type(embedding)} of length {len(embedding)}"
                )
                raise RuntimeError(
                    f"OpenAI returned embedding of wrong dimension: {len(embedding)} vs {EMBED_DIM}"
                )
            
            # Check for NaN values
            embedding_array = np.array(embedding)
            if np.any(np.isnan(embedding_array)):
                LOGGER.error("Embedding contains NaN values")
                raise RuntimeError("OpenAI returned embedding with NaN values")
            
            LOGGER.debug(f"Successfully generated embedding (shape: {len(embedding)})")
            return embedding
        
        except ValueError:
            # Re-raise ValueError as-is
            raise
        except RuntimeError:
            # Re-raise RuntimeError as-is
            raise
        except Exception as e:
            LOGGER.error(f"Unexpected error during API call: {type(e).__name__}: {str(e)}")
            raise RuntimeError(
                f"Failed to generate embedding: {type(e).__name__}: {str(e)}"
            ) from e
    
    def embed_texts_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in a single API call (cost-efficient).
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings, each as list[float] of length 3072
            
        Raises:
            ValueError: If any text is empty
            RuntimeError: If API call fails
            
        Note:
            This is more cost-efficient than calling embed_text multiple times,
            as the API charges per request, not per embedding.
            Future optimization: batch texts client-side if > API limits.
        """
        if not texts or not isinstance(texts, list):
            LOGGER.error(f"Invalid input: texts must be non-empty list, got {type(texts)}")
            raise ValueError("texts must be a non-empty list")
        
        # Validate and trim all texts
        trimmed_texts = []
        for i, text in enumerate(texts):
            if not text or not isinstance(text, str):
                LOGGER.error(f"Text {i} is invalid: {type(text)}")
                raise ValueError(f"Text {i} must be non-empty string")
            trimmed = self._trim_text(text.strip())
            if not trimmed:
                LOGGER.error(f"Text {i} is empty after trimming")
                raise ValueError(f"Text {i} is empty or only whitespace")
            trimmed_texts.append(trimmed)
        
        try:
            LOGGER.debug(f"Calling OpenAI API for {len(trimmed_texts)} embeddings")
            response = self.client.embeddings.create(
                model=self.model,
                input=trimmed_texts,
            )
            
            # Extract embeddings in order
            embeddings = []
            for i, item in enumerate(response.data):
                if item.embedding is None:
                    LOGGER.error(f"Embedding {i} is None")
                    raise RuntimeError(f"OpenAI returned None for embedding {i}")
                
                embedding = item.embedding
                if not isinstance(embedding, list) or len(embedding) != EMBED_DIM:
                    LOGGER.error(
                        f"Embedding {i} has wrong shape: expected {EMBED_DIM}, got {len(embedding)}"
                    )
                    raise RuntimeError(f"Embedding {i} has wrong dimension")
                
                embeddings.append(embedding)
            
            if len(embeddings) != len(trimmed_texts):
                LOGGER.error(
                    f"Returned {len(embeddings)} embeddings for {len(trimmed_texts)} inputs"
                )
                raise RuntimeError("Embedding count mismatch")
            
            LOGGER.debug(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
        
        except ValueError:
            raise
        except RuntimeError:
            raise
        except Exception as e:
            LOGGER.error(f"Unexpected error during batch API call: {type(e).__name__}: {str(e)}")
            raise RuntimeError(
                f"Failed to generate batch embeddings: {type(e).__name__}: {str(e)}"
            ) from e
