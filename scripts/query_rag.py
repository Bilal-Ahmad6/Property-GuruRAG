import argparse
import json
import random
import sys
import time
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer  # type: ignore

# Ensure local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings  # type: ignore

# Simple in-process model cache to avoid reload per query
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}

# Query result cache for performance
_QUERY_CACHE: Dict[str, tuple] = {}
_CACHE_MAX_SIZE = 100

# -----------------------------
# Top-level helper functions (moved out of main so they are reusable
# by answer_one_web and benchmarking utilities)
# -----------------------------

class UserPreferences:
    """Track user preferences across the conversation"""
    def __init__(self):
        self.preferred_locations = []
        self.price_range = {"min": None, "max": None}
        self.property_types = []
        self.size_preferences = []
        self.viewed_properties = []
    
    def update_from_query(self, query: str, analysis: dict):
        """Learn from user interactions"""
        # Track location preferences
        location = analysis.get('filters', {}).get('location')
        if location and location not in self.preferred_locations:
            self.preferred_locations.append(location)
        
        # Track price preferences
        max_price = analysis.get('filters', {}).get('max_price')
        if max_price:
            if not self.price_range["max"] or max_price < self.price_range["max"]:
                self.price_range["max"] = max_price
        
        # Track property type preferences
        prop_type = analysis.get('filters', {}).get('property_type')
        if prop_type and prop_type not in self.property_types:
            self.property_types.append(prop_type)
    
    def get_personalized_filters(self) -> dict:
        """Get filters based on learned preferences"""
        filters = {}
        if self.preferred_locations:
            filters['preferred_locations'] = self.preferred_locations
        if self.price_range["max"]:
            filters['max_price'] = self.price_range["max"]
        if self.property_types:
            filters['property_types'] = self.property_types
        return filters


def get_similar_properties(property_id: str, processed_map: Dict[str, dict], top_n: int = 3) -> List[dict]:
    """Suggest similar properties based on features"""
    if property_id not in processed_map:
        return []
    
    target_property = processed_map[property_id]
    target_price = target_property.get('price_numeric', 0)
    target_bedrooms = target_property.get('bedrooms', '')
    
    similar_properties = []
    
    for lid, prop in processed_map.items():
        if lid == property_id:
            continue
        
        # Calculate similarity score
        score = 0
        prop_price = prop.get('price_numeric', 0)
        
        # Price similarity (within 20% range)
        if target_price > 0 and prop_price > 0:
            price_diff = abs(target_price - prop_price) / target_price
            if price_diff <= 0.2:
                score += 3
            elif price_diff <= 0.5:
                score += 2
        
        # Bedroom similarity
        if target_bedrooms and prop.get('bedrooms') == target_bedrooms:
            score += 2
        
        # Location similarity (same general area)
        target_location = target_property.get('location', '').lower()
        prop_location = prop.get('location', '').lower()
        if target_location and prop_location and target_location in prop_location:
            score += 1
        
        if score > 0:
            similar_properties.append((score, lid, prop))
    
    # Sort by score and return top_n
    similar_properties.sort(key=lambda x: x[0], reverse=True)
    return [{"listing_id": lid, **prop} for _, lid, prop in similar_properties[:top_n]]


def validate_response_quality(query: str, response: str, properties: List[dict]) -> dict:
    """Score response quality and suggest improvements"""
    scores = {}
    
    # Relevance: Do properties match the query intent?
    analysis = enhanced_query_analysis(query)
    filters = analysis.get('filters', {})
    
    relevant_count = 0
    for prop in properties:
        is_relevant = True
        
        # Check price filter
        if 'max_price' in filters:
            prop_price = prop.get('price_numeric', 0)
            if prop_price > filters['max_price'] * 10000000:  # Convert crore to PKR
                is_relevant = False
        
        # Check bedroom filter
        if 'bedrooms' in filters:
            prop_bedrooms = str(prop.get('bedrooms', '')).strip()
            if prop_bedrooms and str(filters['bedrooms']) not in prop_bedrooms:
                is_relevant = False
        
        # Check property type filter (strict matching)
        if 'property_type' in filters:
            prop_title = prop.get('title', '').lower()
            requested_type = filters['property_type'].lower()
            
            if requested_type == 'house':
                # For house requests, exclude apartments/flats
                if any(word in prop_title for word in ['apartment', 'flat', 'unit']):
                    is_relevant = False
                # Must contain house-related terms
                if not any(word in prop_title for word in ['house', 'home', 'villa', 'bungalow', 'marla']):
                    is_relevant = False
            elif requested_type == 'apartment':
                # For apartment requests, exclude houses
                if any(word in prop_title for word in ['house', 'home', 'villa', 'bungalow', 'marla']):
                    is_relevant = False
                # Must contain apartment-related terms
                if not any(word in prop_title for word in ['apartment', 'flat', 'unit']):
                    is_relevant = False
        
        if is_relevant:
            relevant_count += 1
    
    scores['relevance'] = relevant_count / len(properties) if properties else 0
    
    # Completeness: Does response address the query?
    completeness = 0.5  # Base score
    if properties:
        completeness += 0.3
    if len(response.split()) > 10:  # Detailed response
        completeness += 0.2
    scores['completeness'] = min(completeness, 1.0)
    
    # Accuracy: Are property details correct?
    accuracy = 1.0  # Assume accurate unless we detect issues
    for prop in properties:
        if not prop.get('title') or prop.get('title') == 'No title available':
            accuracy -= 0.1
    scores['accuracy'] = max(accuracy, 0.0)
    
    # Overall quality score
    overall_score = sum(scores.values()) / len(scores)
    
    return {
        "scores": scores,
        "overall": overall_score,
        "suggestions": generate_improvement_suggestions(analysis, scores)
    }


def generate_improvement_suggestions(analysis: dict, scores: dict) -> List[str]:
    """Generate suggestions to improve response quality"""
    suggestions = []
    
    if scores.get('relevance', 1) < 0.7:
        suggestions.append("Consider refining search criteria to find more relevant properties")
    
    if scores.get('completeness', 1) < 0.7:
        suggestions.append("Provide more detailed property information")
    
    if analysis['intent'] == 'compare' and len(analysis.get('filters', {})) < 2:
        suggestions.append("Specify comparison criteria (price, location, size, etc.)")
    
    return suggestions


def graceful_error_handling(func):
    """Better error messages and fallback strategies"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_type = type(e).__name__
            if "chroma" in str(e).lower() or "database" in str(e).lower():
                return "I'm having trouble accessing the property database. Please try again in a moment."
            elif "embedding" in str(e).lower() or "model" in str(e).lower():
                return "I couldn't understand your query properly. Could you rephrase it or be more specific?"
            elif "api" in str(e).lower() or "groq" in str(e).lower():
                return "I'm experiencing connectivity issues. Please try your request again."
            else:
                return f"I encountered an unexpected issue. Please try rephrasing your question."
    return wrapper


@graceful_error_handling
def enhanced_retrieve(collection, query: str, k: int, embedding_model: str, user_prefs: UserPreferences = None, filters: dict = None) -> Dict[str, List]:
    """Enhanced retrieval with user preferences, filters, and error handling"""
    # Use the new advanced retrieval by default
    return enhanced_retrieve_v2(collection, query, k, embedding_model, user_prefs, filters)


def is_casual_greeting_or_irrelevant(query: str) -> bool:
    """Return True if the query is a casual greeting or clearly not about real estate."""
    query_lower = query.lower().strip()

    greetings = {
        "hi", "hello", "hey", "good morning", "good evening", "good afternoon",
        "how are you", "what's up", "sup", "yo", "greetings"
    }

    if len(query_lower) <= 3 and query_lower in {"hi", "hey", "yo", "sup"}:
        return True
    for greeting in greetings:
        if query_lower == greeting or query_lower.startswith(greeting):
            return True

    irrelevant_keywords = {
        "weather", "food", "music", "movie", "game", "sport", "politics",
        "recipe", "joke", "funny", "meme", "cat", "dog", "animal"
    }
    words = query_lower.split()
    if any(k in words for k in irrelevant_keywords):
        return True
    return False


def get_casual_response(query: str) -> str:
    query_lower = query.lower().strip()
    if any(greet in query_lower for greet in ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]):
        return (
            "Hello! I'm your real estate assistant for Bahria Town Phase 7. "
            "I can help you find properties based on your preferences.\n\n"
            "Try asking me something like:\n"
            "- Show me 10 marla houses under 50000000\n"
            "- Find apartments with 2 bedrooms\n"
            "- What's available near parks?"
        )
    if "how are you" in query_lower:
        return (
            "I'm doing great, thank you for asking! I'm here and ready to help you find the perfect property in Bahria Town Phase 7.\n\n"
            "What kind of property are you looking for today?"
        )
    return (
        "I'm a specialized real estate assistant for Bahria Town Phase 7 properties. "
        "I can help you search for houses, apartments, and other properties based on your criteria.\n\n"
        "Could you please ask me something related to real estate? For example:\n"
        "- Property size (marla, sqft)\n"
        "- Price range\n"
        "- Number of bedrooms/bathrooms\n"
        "- Location preferences"
    )


def enhanced_query_analysis(query: str) -> dict:
    """Enhanced intent detection and query parsing"""
    query_lower = query.lower()
    
    # Detect intent
    intent = "search"  # default
    if any(word in query_lower for word in ["compare", "vs", "versus", "difference"]):
        intent = "compare"
    elif any(word in query_lower for word in ["average", "mean", "total", "count", "how many"]):
        intent = "analyze"
    elif any(word in query_lower for word in ["similar", "like", "same as"]):
        intent = "recommend"
    
    # Extract filters
    filters = {}
    
    # Price extraction
    import re
    price_patterns = [
        r'under (\d+(?:\.\d+)?)\s*(?:crore|lakh|million)',
        r'below (\d+(?:\.\d+)?)\s*(?:crore|lakh|million)',
        r'less than (\d+(?:\.\d+)?)\s*(?:crore|lakh|million)',
        r'(\d+(?:\.\d+)?)\s*(?:crore|lakh|million)\s*(?:or less|max|maximum)',
    ]
    for pattern in price_patterns:
        match = re.search(pattern, query_lower)
        if match:
            filters['max_price'] = float(match.group(1))
            break
    
    # Bedroom extraction
    bedroom_match = re.search(r'(\d+)\s*(?:bed|bedroom)', query_lower)
    if bedroom_match:
        filters['bedrooms'] = int(bedroom_match.group(1))
    
    # Area size extraction
    area_match = re.search(r'(\d+)\s*(?:marla|sqft|square feet)', query_lower)
    if area_match:
        filters['area_size'] = int(area_match.group(1))
    
    # Property type extraction
    property_type = None
    if any(word in query_lower for word in ["house", "home", "villa", "bungalow"]):
        property_type = "house"
    elif any(word in query_lower for word in ["apartment", "flat", "unit"]):
        property_type = "apartment"
    
    if property_type:
        filters['property_type'] = property_type
    
    # Urgency detection
    urgency = "normal"
    if any(word in query_lower for word in ["urgent", "asap", "immediately", "quickly"]):
        urgency = "high"
    
    return {
        "intent": intent,
        "filters": filters,
        "urgency": urgency,
        "original_query": query
    }


def extract_requested_number(query: str) -> Optional[int]:
    """Extract the number of properties requested from the query."""
    query_lower = query.lower().strip()
    
    # Check for written numbers first
    word_to_number = {
        'one': 1, 'a': 1, 'single': 1, 'an': 1,
        'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    
    # Look for written numbers with property keywords
    import re
    for word, num in word_to_number.items():
        patterns = [
            rf'\b{word}\s+(?:house|apartment|property|listing|properties|homes|units|place|option)\b',
            rf'\b{word}\s+(?:bedroom|bed|br)\b',  # "one bedroom"
            rf'\bshow\s+me\s+{word}\b',  # "show me one"
            rf'\bfind\s+{word}\b',  # "find one"
            rf'\bgive\s+me\s+{word}\b',  # "give me one"
            rf'\bwant\s+{word}\b',  # "want two"
            rf'\bneed\s+{word}\b',  # "need one"
        ]
        for pattern in patterns:
            if re.search(pattern, query_lower):
                return num
    
    # Check for digit numbers
    number_match = re.search(r'\b(\d+)\s*(?:house|apartment|property|listing|properties|homes|units)', query_lower)
    if number_match:
        return int(number_match.group(1))
    
    # Special cases for single property requests
    single_indicators = [
        r'\ba\s+(?:house|apartment|property|home|place)',
        r'\ban\s+(?:apartment|option)',
        r'\bshow\s+me\s+(?:a|an)\b',
        r'\bfind\s+(?:a|an)\b',
        r'\bany\s+(?:house|apartment|property)\b'
    ]
    
    for pattern in single_indicators:
        if re.search(pattern, query_lower):
            return 1
    
    return None


def should_use_context(current_query: str, history: List[dict]) -> bool:
    """Intelligent decision on when to use conversation history"""
    if not history:
        return False
    
    query_lower = current_query.lower().strip()
    
    # Use context for pronouns and references
    pronouns = ["it", "that", "this", "them", "those", "these"]
    if any(pronoun in query_lower.split() for pronoun in pronouns):
        return True
    
    # Use context for comparative queries
    comparatives = ["cheaper", "bigger", "smaller", "better", "similar", "like that", "more"]
    if any(comp in query_lower for comp in comparatives):
        return True
    
    # Use context for follow-up location queries (very short)
    location_words = ["in", "at", "near", "around", "from"]
    if len(query_lower.split()) <= 2 and any(word in query_lower for word in location_words):
        return True
    
    # Use context for continuation words
    continuation_words = ["more", "another", "also", "additionally", "plus"]
    if any(word in query_lower for word in continuation_words):
        return True
    
    return False


def normalize_location_query(query: str) -> str:
    """Normalize location terms to improve matching."""
    query_lower = query.lower()
    
    # Location normalization mappings
    location_mappings = {
        'river hills': 'river hill',
        'riverhills': 'river hill',
        'spring north': 'bahria spring north',
        'springnorth': 'bahria spring north',
        'spring south': 'bahria spring south',
        'springsouth': 'bahria spring south',
        'phase 7': 'bahria town phase 7',
        'phase7': 'bahria town phase 7',
        'p7': 'phase 7',
        'bt': 'bahria town',
        'bahria phase 7': 'bahria town phase 7',
        # Add variations for better matching
        ' hills ': ' hill ',
        ' north ': ' north ',
        ' south ': ' south ',
    }
    
    normalized = query_lower
    for variant, canonical in location_mappings.items():
        if variant in normalized:
            normalized = normalized.replace(variant, canonical)
    
    # Also create an expanded query with location variations for better embedding match
    location_expansions = []
    if 'river hill' in normalized:
        location_expansions.extend(['river hills', 'riverhills', 'river hill'])
    if 'spring north' in normalized:
        location_expansions.extend(['spring north', 'springnorth', 'bahria spring north'])
    if 'spring south' in normalized:
        location_expansions.extend(['spring south', 'springsouth', 'bahria spring south'])
    
    if location_expansions:
        # Add variations to help with embedding matching
        normalized = f"{normalized} {' '.join(location_expansions)}"
    
    return normalized


def is_statistical_query(query: str) -> bool:
    query_lower = query.lower()
    statistical_keywords = [
        "average", "mean", "avg", "typical", "usual", "common",
        "total", "sum", "count", "how many", "number of",
        "cheapest", "most expensive", "highest", "lowest",
        "price range", "cost range", "compare prices"
    ]
    return any(keyword in query_lower for keyword in statistical_keywords)
    query_lower = query.lower()
    statistical_keywords = [
        "average", "mean", "avg", "typical", "usual", "common",
        "total", "sum", "count", "how many", "number of",
        "cheapest", "most expensive", "highest", "lowest",
        "price range", "cost range", "compare prices"
    ]
    return any(keyword in query_lower for keyword in statistical_keywords)


def get_all_property_data_for_analysis(processed_map: Dict[str, dict], query: str) -> List[dict]:
    query_lower = query.lower()
    all_properties: List[dict] = []
    for listing_id, data in processed_map.items():
        property_info = {
            "listing_id": listing_id,
            "title": data.get("title", ""),
            "price_numeric": data.get("price_numeric", 0),
            "price_raw": data.get("price_raw", ""),
            "bedrooms": data.get("bedrooms", ""),
            "bathrooms": data.get("bathrooms", ""),
            "area_unit": data.get("area_unit", ""),
            "area_size": data.get("area_size", ""),
            "url": data.get("url", ""),
            "scraped_at": data.get("processed_at", ""),
        }
        include_property = True
        if "10 marla" in query_lower and "marla" in str(data.get("area_unit", "")).lower():
            include_property = True
        elif "marla" in query_lower and "marla" not in str(data.get("area_unit", "")).lower():
            include_property = False
        if "house" in query_lower and "apartment" in str(data.get("title", "")).lower():
            include_property = False
        elif "apartment" in query_lower and "house" in str(data.get("title", "")).lower():
            include_property = False
        if include_property and property_info["price_numeric"] and property_info["price_numeric"] > 0:
            all_properties.append(property_info)
    return all_properties


def has_relevant_property_results(res: Dict[str, List], threshold: float = 1.2) -> bool:
    distances = res.get("distances", [[]])[0]
    if not distances:
        return False
    best_distance = min(distances) if distances else 1.0
    return best_distance < threshold

# Web API function for Flask integration
def answer_one_web(query: str, groq_api_key: str, groq_model: str = "llama3-8b-8192") -> str:
    """Simplified web-compatible answer function (string only)."""
    result = rag_infer(query=query, groq_api_key=groq_api_key, groq_model=groq_model)
    return result.get("answer", "")


def rag_infer(
    query: str,
    groq_api_key: str = None,
    cohere_api_key: str = None,
    llm_engine: str = "cohere",  # Default to cohere
    groq_model: str = "llama-3.1-70b-versatile",
    cohere_model: str = "command-r-plus",
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    k: int = 5,
    conversation_history: List[dict] = None,
) -> Dict[str, object]:
    """Structured RAG inference used by web + benchmarking.

    Returns a dict containing: answer, retrieved (list of dict with listing_id, title, distance),
    requested_number, used_engine, context (truncated), mode (greeting|stats|retrieval|general_error).
    """
    if conversation_history is None:
        conversation_history = []
        
    out: Dict[str, object] = {"query": query}
    try:
        script_dir = Path(__file__).parent.parent
        processed_path = script_dir / "data" / "processed" / "zameen_phase7_processed.json"
        processed_map = load_processed_map(processed_path)
        out["processed_loaded"] = bool(processed_map)
        if not processed_map:
            out["answer"] = "Sorry, I couldn't load the property database. Please try again later."
            out["mode"] = "general_error"
            return out

        if is_casual_greeting_or_irrelevant(query):
            out["answer"] = get_casual_response(query)
            out["mode"] = "greeting"
            return out

        if is_statistical_query(query):
            all_properties = get_all_property_data_for_analysis(processed_map, query)
            if not all_properties:
                out["answer"] = "Sorry, I couldn't analyze the property data. Please try again later."
                out["mode"] = "stats"
                return out
            analysis_prompt = f"""You are analyzing real estate data for Bahria Town Phase 7. Here is the complete property dataset:\n\n{json.dumps(all_properties[:50], indent=2)}\n\nUser Question: {query}\n\nProvide a detailed analysis with:\n1. Direct answer to their question (with calculated numbers)\n2. Key statistics (average, range, etc.)\n3. Notable insights\n4. Be specific and use the actual data provided."""
            
            # Determine which API key and model to use
            if llm_engine.lower() == "cohere" and cohere_api_key:
                out["answer"] = call_llm(analysis_prompt, "cohere", cohere_model, cohere_api_key, 500)
            elif llm_engine.lower() == "groq" and groq_api_key:
                out["answer"] = call_llm(analysis_prompt, "groq", groq_model, groq_api_key, 500)
            else:
                out["answer"] = "API key missing for the selected LLM engine."
            
            out["mode"] = "stats"
            return out

        # Retrieval flow
        try:
            phase_start = time.perf_counter()
            chroma_client = chromadb.PersistentClient(path=str(script_dir / "chromadb_data"))
            
            # Normalize location terms and extract numbers
            normalized_query = normalize_location_query(query)
            requested_number = extract_requested_number(query)
            
            # Check if this is a follow-up question that needs context expansion
            expanded_query = normalized_query
            if conversation_history and len(normalized_query.strip().split()) <= 2:
                # Only expand very short queries that might need context (like "in river hill")
                last_exchange = conversation_history[-1] if conversation_history else None
                if last_exchange and any(word in normalized_query.lower() for word in ['in', 'at', 'near', 'from']):
                    # Only expand if it's a location-based query
                    expanded_query = f"{last_exchange['user']} {normalized_query}"
            
            # Embedding timing
            t0 = time.perf_counter()
            embedding = embed_query(embedding_model, expanded_query)
            t1 = time.perf_counter()
            collection = chroma_client.get_collection("zameen_listings")
            res = collection.query(query_embeddings=[embedding], n_results=k)
            t2 = time.perf_counter()
            out["raw_retrieval"] = {
                "distances": res.get("distances", [[]])[0],
                "ids": res.get("ids", [[]])[0],
            }
            if not has_relevant_property_results(res):
                # Include conversation history in general chat
                conversation_context = ""
                if conversation_history:
                    conversation_context = "Previous conversation:\n"
                    for exchange in conversation_history[-3:]:  # Last 3 exchanges
                        conversation_context += f"User: {exchange['user']}\n"
                        conversation_context += f"Assistant: {exchange['assistant']}\n\n"
                    conversation_context += "Current query:\n"
                
                general_prompt = (
                    f"{conversation_context}You are a friendly real estate assistant for Bahria Town Phase 7. The user asked: '{query}'.\n"
                    "I couldn't find specific matching properties in the database. Acknowledge politely, suggest a more specific search (price range, bedrooms, area size) and offer general info. Consider the conversation context."
                )
                gen_start = time.perf_counter()
                
                # Use the appropriate LLM engine
                if llm_engine.lower() == "cohere" and cohere_api_key:
                    out["answer"] = call_llm(general_prompt, "cohere", cohere_model, cohere_api_key)
                elif llm_engine.lower() == "groq" and groq_api_key:
                    out["answer"] = call_llm(general_prompt, "groq", groq_model, groq_api_key)
                else:
                    out["answer"] = "API key missing for the selected LLM engine."
                gen_end = time.perf_counter()
                out["timings"] = {
                    "embed_ms": (t1 - t0) * 1000.0,
                    "retrieve_ms": (t2 - t1) * 1000.0,
                    "generate_ms": (gen_end - gen_start) * 1000.0,
                    "total_ms": (gen_end - phase_start) * 1000.0,
                }
                out["mode"] = "no_retrieval"
                return out

            actual_top_n = requested_number if requested_number else 3  # Use exact number requested, default to 3
            context, listings = build_context(res, processed_map, top_n=actual_top_n)
            
            # Build conversation context - only for very short queries that need context
            conversation_context = ""
            if conversation_history and len(query.strip().split()) <= 2:
                # Only add context for very short queries like "in river hill", "show more", etc.
                conversation_context = "\nRECENT CONTEXT (for reference only):\n"
                for exchange in conversation_history[-1:]:  # Only last exchange
                    conversation_context += f"Previous: {exchange['user']}\n"
                conversation_context += "END CONTEXT\n\n"
            
            system = (
                
                "You are a conversational real-estate assistant for Bahria Town Phase 7. CRITICAL RULES:\n"
                "1. Use ONLY the information provided in the context below - NO external knowledge\n"
                "2. Provide ONLY a brief conversational response - do NOT list property details\n"
                "3. Property listings will be shown separately in a formatted table\n"
                "4. If no relevant properties found, say 'No matching properties found in the database'\n"
                "5. If properties are found, say something like 'I found X properties matching your criteria'\n"
                "6. Do NOT format or list individual properties - just provide conversational context\n"
                "7. STRICTLY follow the user's requested number - if they ask for 1, show only 1; if they ask for 3, show exactly 3\n"
                "8. Each new query is independent - respond to what the user is currently asking\n"
                "9. Do NOT question or compare with previous requests - just answer the current query\n"
                "10. Keep your response brief and conversational\n"
                "11. If the user asks for a specific number of properties, return exactly that many, no more, no less.\n"
                "12. If the user asks for a specific type of property, ensure the response matches that type.\n"
                "13. If the user asks for a specific location, ensure the response is relevant to that location.\n"
                "14. If the user asks for a specific price range, ensure the response matches that range.\n"
                "15. If the user asks for a specific number of bedrooms, ensure the response matches that criteria.\n"
                "16. If the user's requirements are not met in the dataset, give the closest matching properties.\n"
                "17. If it is totally irrelevant, don't mention no data found, instead just list a property randomly, without any context.\n"
                "18. Be EXTRA CAUTIOUS about listing house and apartment, list what the user asks exactly.\n"
                "19. The search should be mainly from the title of the properties and the price.\n"
                "20. If the user asks for a specific location, ensure the response is relevant to that location according to the title of the property.\n"
                "21. Properties should be listed in order of relevance, with the most relevant properties shown first, and the relevance is according to the words in the title and the price if the user asks with respect to price.\n"
                "22. Never hallucinate property details or invent listings not present in the context.\n"
                "23. If the user asks for amenities or features, only mention them if present in the context/title.\n"
                "24. If the user query is ambiguous, politely ask for clarification instead of guessing.\n"
                "25. If the user asks for both houses and apartments, clarify that results are filtered strictly by type.\n"
                "26. If the user asks for properties with a certain status (e.g., furnished, new), only mention if present in the context/title.\n"
                "27. Always prioritize properties with exact keyword matches in the title over partial matches.\n"
                "28. If the user asks for the latest or newest properties, prefer those with the most recent scraped_at date.\n"
                "29. Sort matches by Exact keyword match in title\n"
                "30. Search primarily by property title and price.\n"
                
            )
            number_instruction = (
                f"\nThe user specifically asked for {requested_number} properties. Return exactly {requested_number} properties, no more, no less."
                if requested_number else ""
            )
            prompt = (
                f"System: {system}{number_instruction}\n\n"
                f"{conversation_context}"
                f"CONTEXT (use ONLY this information):\n{context}\n\n"
                f"User Query: {query}\n\n"
                f"INSTRUCTIONS:\n"
                f"- Provide ONLY a brief conversational response\n"
                f"- Do NOT list or format property details\n"
                f"- If properties found, just say how many match the criteria\n"
                f"- If no matches, say 'No matching properties found'\n"
                f"- Answer the current query directly - do NOT question the user's request\n"
                f"- Keep response under 2 sentences\n\n"
                "Brief Response:"
            )
            gen_start = time.perf_counter()
            
            # Use the appropriate LLM engine
            if llm_engine.lower() == "cohere" and cohere_api_key:
                answer_text = call_llm(prompt, "cohere", cohere_model, cohere_api_key, 500)
            elif llm_engine.lower() == "groq" and groq_api_key:
                answer_text = call_llm(prompt, "groq", groq_model, groq_api_key, 500)
            else:
                answer_text = "API key missing for the selected LLM engine."
            
            gen_end = time.perf_counter()
            # Prepare retrieval summary for evaluation
            retrieved: List[dict] = []
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]
            for doc, meta, dist in zip(docs, metas, dists):
                if isinstance(meta, dict) and meta.get("listing_id"):
                    lid = meta.get("listing_id")
                    title = processed_map.get(str(lid), {}).get("title")
                    retrieved.append({
                        "listing_id": lid,
                        "title": title,
                        "distance": dist,
                    })
            out.update({
                "answer": answer_text,
                "mode": "retrieval",
                "requested_number": requested_number,
                "retrieved": retrieved,
                "context": context[:1500],
                "timings": {
                    "embed_ms": (t1 - t0) * 1000.0,
                    "retrieve_ms": (t2 - t1) * 1000.0,
                    "generate_ms": (gen_end - gen_start) * 1000.0,
                    "total_ms": (gen_end - phase_start) * 1000.0,
                },
            })
            return out
        except Exception as e:
            out["answer"] = f"Sorry, I encountered an error searching the database: {e}" \
                if "answer" not in out else out["answer"]
            out["mode"] = "retrieval_error"
            return out
    except Exception as e:  # Catch-all
        out["answer"] = f"Sorry, I encountered an unexpected error: {e}"
        out["mode"] = "unexpected_error"
        return out


def choose_device() -> str:
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_processed_map(processed_path: Path) -> Dict[str, dict]:
    if not processed_path.exists():
        return {}
    try:
        with processed_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    out: Dict[str, dict] = {}
    for rec in data:
        lid = rec.get("listing_id")
        if lid:
            out[str(lid)] = rec
    return out


def embed_query(model_name: str, text: str) -> List[float]:
    if model_name in _MODEL_CACHE:
        model = _MODEL_CACHE[model_name]
    else:
        device = choose_device()
        model = SentenceTransformer(model_name, device=device)
        _MODEL_CACHE[model_name] = model
    vec = model.encode([text], batch_size=1, convert_to_numpy=False)[0]
    return vec.tolist() if hasattr(vec, "tolist") else list(vec)


def retrieve(
    collection_name: str, question: str, k: int, embedding_model: str
) -> Dict[str, List]:
    """Retrieve with caching for better performance"""
    # Check cache first
    cache_key = f"{question}_{k}_{embedding_model}"
    if cache_key in _QUERY_CACHE:
        return _QUERY_CACHE[cache_key]
    
    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    try:
        # Prefer getting existing collection to avoid accidental creation with mismatched schemas
        if hasattr(client, "get_collection"):
            collection = client.get_collection(name=collection_name)
        else:
            collection = client.get_or_create_collection(name=collection_name)
        q_emb = embed_query(embedding_model, question)
        res = collection.query(
            query_embeddings=[q_emb], n_results=k, include=["metadatas", "documents", "distances"]
        )
        
        # Cache the result
        if len(_QUERY_CACHE) >= _CACHE_MAX_SIZE:
            # Remove oldest entry
            oldest_key = next(iter(_QUERY_CACHE))
            del _QUERY_CACHE[oldest_key]
        
        _QUERY_CACHE[cache_key] = res
        return res
    except Exception as e:
        print(
            "Warning: retrieval failed (" + str(e) + ").\n"
            "If this is a Chroma schema error, try deleting the 'chromadb_data' folder and re-running embeddings."
        )
        # Return empty result structure
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}


def build_context(res: Dict[str, List], processed_map: Dict[str, dict], top_n: int = 5) -> Tuple[str, List[dict]]:
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    lines: List[str] = []
    seen_listings: set = set()
    picked: List[dict] = []

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        lid = (meta or {}).get("listing_id") if isinstance(meta, dict) else None
        if not lid:
            continue
        # Add chunk line
        chunk_type = (meta or {}).get("chunk_type")
        price = (meta or {}).get("price_numeric")
        br = (meta or {}).get("bedrooms")
        ba = (meta or {}).get("bathrooms")
        au = (meta or {}).get("area_unit")
        url = (meta or {}).get("url")
        title = (processed_map.get(lid, {}) or {}).get("title")
        price_raw = (processed_map.get(lid, {}) or {}).get("price_raw")
        lines.append(
            f"[dist={dist:.4f}] [{chunk_type}] listing_id={lid} | title={title} | price={price_raw or price} | br={br} ba={ba} area_unit={au} | url={url} | text={ (doc or '')[:240] }"
        )

        # Track top unique listings with comprehensive details
        if lid not in seen_listings and len(picked) < top_n:
            seen_listings.add(lid)
            
            # Get comprehensive property details
            processed_listing = processed_map.get(lid, {})
            
            picked.append({
                "listing_id": lid,
                "title": title or "No title available",
                "price": price_raw or price or "Price not available",
                "location": processed_listing.get("location", "Bahria Town Phase 7"),
                "url": url or "URL not available",
                "scraped_at": ((processed_listing.get("raw", {}) or {}).get("scraped_at")
                or processed_listing.get("processed_at", "Unknown")),
            })

    # Don't shuffle when user requests specific number - maintain relevance order
    # random.shuffle(picked)  # Removed to respect user's specific requests
    
    context = "\n".join(lines[: top_n * 2])  # include up to 2 chunks per listing
    return context, picked


def call_llm_llamacpp(prompt: str, model_path: str, max_tokens: int = 384) -> str:
    try:
        from llama_cpp import Llama  # type: ignore
    except Exception:
        return "[LLM unavailable: please install llama-cpp-python and provide a GGUF model path]"
    llm = Llama(model_path=model_path, n_ctx=4096, n_threads=6)
    out = llm(prompt=prompt, max_tokens=max_tokens, temperature=0.2)
    return out.get("choices", [{}])[0].get("text", "").strip()


def call_llm_gpt4all(prompt: str, model_name: str, max_tokens: int = 384) -> str:
    try:
        from gpt4all import GPT4All  # type: ignore
    except Exception:
        return "[LLM unavailable: please install gpt4all and provide a local model name]"
    gpt = GPT4All(model_name)
    with gpt.chat_session():
        return gpt.generate(prompt, max_tokens=max_tokens, temp=0.2)


def call_llm_ollama(prompt: str, model_name: str, max_tokens: int = 384) -> str:
    try:
        import requests
    except Exception:
        return "[LLM unavailable: please install requests library]"
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.2
                },
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        return f"[Ollama unavailable: {str(e)}]"


def call_llm_groq(prompt: str, model_name: str, api_key: str, max_tokens: int = 384) -> str:
    try:
        import requests
    except Exception:
        return "[LLM unavailable: please install requests library]"
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.1  # Lower temperature for more deterministic, factual responses
            },
            timeout=30
        )
        
        if not response.ok:
            # Get more detailed error info
            try:
                error_detail = response.json()
                return f"[Groq API Error: {response.status_code} - {error_detail}]"
            except:
                return f"[Groq API Error: {response.status_code} - {response.text}]"
        
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[Groq unavailable: {str(e)}]"


def call_llm_cohere(prompt: str, model_name: str, api_key: str, max_tokens: int = 384) -> str:
    try:
        import requests
    except Exception:
        return "[LLM unavailable: please install requests library]"
    
    try:
        response = requests.post(
            "https://api.cohere.ai/v1/generate",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.1,  # Lower temperature for more deterministic responses
                "truncate": "END"
            },
            timeout=30
        )
        
        if not response.ok:
            try:
                error_detail = response.json()
                return f"[Cohere API Error: {response.status_code} - {error_detail}]"
            except:
                return f"[Cohere API Error: {response.status_code} - {response.text}]"
        
        result = response.json()
        return result["generations"][0]["text"].strip()
    except Exception as e:
        return f"[Cohere unavailable: {str(e)}]"


def call_llm(prompt: str, engine: str, model_name: str, api_key: str, max_tokens: int = 384) -> str:
    """Unified LLM calling function that supports multiple engines."""
    if engine.lower() == "cohere":
        return call_llm_cohere(prompt, model_name, api_key, max_tokens)
    elif engine.lower() == "groq":
        return call_llm_groq(prompt, model_name, api_key, max_tokens)
    else:
        return f"[Unsupported LLM engine: {engine}]"


def format_answer(question: str, listings: List[dict], freshness_note: str, answer_text: str) -> str:
    """Format the answer with clean property details."""
    
    if not listings:
        return f"\n{answer_text}\n\nNo matching properties found.\n\n{freshness_note}"
    
    # Build detailed property listings with clean formatting
    lines = [f"\n{answer_text}\n"]
    lines.append("MATCHING PROPERTIES:")
    lines.append("=" * 60)
    
    for i, listing in enumerate(listings, 1):
        title = listing.get('title', 'No title available')
        price = listing.get('price', 'Price not available')
        url = listing.get('url', 'URL not available')
        scraped_at = listing.get('scraped_at', 'Unknown')
        
        lines.append(f"\nProperty #{i}")
        lines.append(f"Title: {title}")
        lines.append(f"Price: {price}")
        
        # Property specifications
        lines.append("Specifications:")
        bedrooms = listing.get('bedrooms', 'Not specified')
        bathrooms = listing.get('bathrooms', 'Not specified') 
        area_unit = listing.get('area_unit', 'Not specified')
        area_size = listing.get('area_size', 'Not specified')
        property_type = listing.get('property_type', 'Not specified')
        location = listing.get('location', 'Bahria Town Phase 7')
        
        lines.append(f"  Bedrooms: {bedrooms}")
        lines.append(f"  Bathrooms: {bathrooms}")
        lines.append(f"  Area: {area_size} {area_unit}")
        lines.append(f"  Type: {property_type}")
        lines.append(f"  Location: {location}")
        
        lines.append(f"Link: {url}")
        lines.append(f"Data Date: {scraped_at}")
        
        if i < len(listings):
            lines.append("-" * 60)
    
    lines.append(f"\n{freshness_note}")
    
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG query using Chroma + local LLM")
    parser.add_argument("--collection", default=settings.collection_name)
    parser.add_argument("--query")
    parser.add_argument("--k", type=int, default=5, help="Top-N chunks to retrieve")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformers model for query embedding",
    )
    parser.add_argument("--processed", type=Path, default=Path("data/processed/zameen_phase7_processed.json"))
    parser.add_argument("--llm-engine", choices=["llama", "gpt4all", "ollama", "groq", "cohere", "none"], default="none")
    parser.add_argument("--llama-model-path", help="Path to GGUF model for llama.cpp", default=None)
    parser.add_argument("--gpt4all-model", help="Local GPT4All model name", default="orca-mini-3b.gguf2.Q4_0.gguf")
    parser.add_argument("--ollama-model", help="Ollama model name", default="llama3.1:8b-instruct-q4_K_M")
    parser.add_argument("--groq-api-key", help="Groq API key", default=None)
    parser.add_argument("--groq-model", help="Groq model name", default="llama3-8b-8192")
    parser.add_argument("--cohere-api-key", help="Cohere API key", default=None)
    parser.add_argument("--cohere-model", help="Cohere model name", default="command-r-plus")
    parser.add_argument("--explain", action="store_true", help="Print retrieved chunks and scores")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat loop")
    args = parser.parse_args()

    processed_map = load_processed_map(args.processed)

    # (helpers now imported from top-level definitions above)
    
    def call_llm_for_general_chat(prompt: str) -> str:
        """Call LLM for general conversation without property context."""
        if args.llm_engine == "llama" and args.llama_model_path:
            return call_llm_llamacpp(prompt, args.llama_model_path)
        elif args.llm_engine == "gpt4all":
            return call_llm_gpt4all(prompt, args.gpt4all_model)
        elif args.llm_engine == "ollama":
            return call_llm_ollama(prompt, args.ollama_model)
        elif args.llm_engine == "groq" and args.groq_api_key:
            return call_llm_groq(prompt, args.groq_model, args.groq_api_key)
        elif args.llm_engine == "cohere" and args.cohere_api_key:
            return call_llm_cohere(prompt, args.cohere_model, args.cohere_api_key)
        else:
            return "I'm a real estate assistant. For general questions, please enable an LLM engine (cohere, groq, ollama, gpt4all, or llama)."

    def answer_one(q: str, conversation_history: List[dict] = None) -> str:
        if conversation_history is None:
            conversation_history = []
        
        # Initialize user preferences (in a real app, this would be persistent)
        user_prefs = UserPreferences()
        
        # Enhanced query analysis
        analysis = enhanced_query_analysis(q)
        user_prefs.update_from_query(q, analysis)
            
        # Handle obvious greetings first
        if is_casual_greeting_or_irrelevant(q):
            casual_response = get_casual_response(q)
            return casual_response
        
        # Check if this is a statistical/analytical query
        if is_statistical_query(q):
            # Get all property data for analysis
            all_properties = get_all_property_data_for_analysis(processed_map, q)
            
            if all_properties:
                # Create context for statistical analysis
                stats_context = f"Property Data Analysis for query: '{q}'\n\n"
                stats_context += "Available Properties:\n"
                
                for i, prop in enumerate(all_properties[:50], 1):  # Limit to 50 for context
                    stats_context += f"{i}. {prop['title']} - PKR {prop['price_numeric']:,} - {prop['area_size']} {prop['area_unit']}\n"
                
                stats_context += f"\nTotal properties found: {len(all_properties)}"
                
                # Use LLM to analyze the data
                analysis_prompt = f"""You are a real estate data analyst. Based on the property data provided, answer the user's question with specific numbers and calculations.

Data: {stats_context}

User Question: {q}

Provide a detailed analysis with:
1. Direct answer to their question (with calculated numbers)
2. Key statistics (average, range, etc.)
3. Notable insights
4. Be specific and use the actual data provided."""

                if args.llm_engine != "none":
                    answer_text = call_llm_for_general_chat(analysis_prompt)
                    return answer_text
                else:
                    # Fallback: basic statistical analysis
                    prices = [p['price_numeric'] for p in all_properties if p['price_numeric'] > 0]
                    if prices:
                        avg_price = sum(prices) / len(prices)
                        min_price = min(prices)
                        max_price = max(prices)
                        answer_text = f"Based on {len(all_properties)} properties:\n"
                        answer_text += f"- Average price: PKR {avg_price:,.0f}\n"
                        answer_text += f"- Price range: PKR {min_price:,} to PKR {max_price:,}\n"
                        answer_text += f"- Total properties analyzed: {len(all_properties)}"
                        return answer_text
                    else:
                        return "No valid price data found for analysis."
            else:
                return "No properties found matching your criteria for analysis."
        
        # Regular property search (non-statistical)
        # Normalize location terms and extract numbers
        normalized_query = normalize_location_query(q)
        requested_number = extract_requested_number(q)
        
        # Check if this is a follow-up question that needs context expansion
        expanded_query = normalized_query
        if should_use_context(q, conversation_history):
            last_exchange = conversation_history[-1] if conversation_history else None
            if last_exchange:
                # Combine with previous query for better retrieval
                expanded_query = f"{last_exchange['user']} {normalized_query}"
        
        # Use enhanced retrieval with error handling
        res = enhanced_retrieve(args.collection, expanded_query, args.k, args.embedding_model, user_prefs, analysis.get('filters', {}))

        # Check if we found relevant property results
        if has_relevant_property_results(res):
            # Adjust top_n based on user request
            if requested_number:
                actual_top_n = min(requested_number, 5)  # Cap at 5 for performance
            else:
                # For general queries, let LLM decide from context but limit retrieval to 3 for better relevance
                actual_top_n = 3
            
            context, listings = build_context(res, processed_map, top_n=actual_top_n)
            
            # Apply strict property type filtering
            if analysis.get('filters', {}).get('property_type'):
                filtered_listings = []
                requested_type = analysis['filters']['property_type'].lower()
                
                for listing in listings:
                    title = listing.get('title', '').lower()
                    include_listing = True
                    
                    if requested_type == 'house':
                        # For house requests, exclude apartments/flats
                        if any(word in title for word in ['apartment', 'flat', 'unit']):
                            include_listing = False
                        # Must contain house-related terms
                        elif not any(word in title for word in ['house', 'home', 'villa', 'bungalow', 'marla']):
                            include_listing = False
                    elif requested_type == 'apartment':
                        # For apartment requests, exclude houses
                        if any(word in title for word in ['house', 'home', 'villa', 'bungalow', 'marla']):
                            include_listing = False
                        # Must contain apartment-related terms
                        elif not any(word in title for word in ['apartment', 'flat', 'unit']):
                            include_listing = False
                    
                    if include_listing:
                        filtered_listings.append(listing)
                
                listings = filtered_listings
                
                # If no listings match the strict filter, inform the user
                if not listings:
                    property_type_name = "houses" if requested_type == 'house' else "apartments/flats"
                    return f"No {property_type_name} found matching your criteria in the database. Try adjusting your search parameters."
            
            if args.explain:
                print("Retrieved chunks (debug):\n")
                print(context)
            
            # Enhanced system prompt that enforces strict adherence to context
            system = (
                "You are a conversational real-estate assistant. CRITICAL RULES:\n"
                "1. Use ONLY information from the context below - NO external knowledge\n"
                "2. Provide ONLY a brief conversational response - do NOT list property details\n"
                "3. Property listings will be shown separately in a formatted table\n"
                "4. If no relevant properties found, say 'No matching properties found in the database'\n"
                "5. If properties are found, say something like 'I found X properties matching your criteria'\n"
                "6. Do NOT format or list individual properties - just provide conversational context\n"
                "7. Each new query is independent - respond to what the user is currently asking\n"
                "8. Do NOT question or compare with previous requests - just answer the current query\n"
                "9. Keep your response brief and conversational\n"
                "10. If user asks for recommendations or similar properties, mention that feature\n"
                "11. STRICT PROPERTY TYPE FILTERING: Houses and apartments are completely separate - never mix them"
            )
            
            # Build conversation context - only for queries that truly need it
            conversation_context = ""
            if should_use_context(q, conversation_history):
                conversation_context = "\nRECENT CONTEXT (for reference only):\n"
                for exchange in conversation_history[-1:]:  # Only last exchange
                    conversation_context += f"Previous: {exchange['user']}\n"
                conversation_context += "END CONTEXT\n\n"
            
            # Add specific instruction about number if detected
            number_instruction = ""
            if requested_number:
                number_instruction = f"\nThe user asked for {requested_number} properties. Mention this in your response."
            
            # Add property type instruction
            property_type_instruction = ""
            if analysis.get('filters', {}).get('property_type'):
                prop_type = analysis['filters']['property_type']
                property_type_instruction = f"\nIMPORTANT: User specifically asked for {prop_type}s - results are strictly filtered for this property type only."
            
            prompt = (
                f"System: {system}{number_instruction}{property_type_instruction}\n\n"
                f"{conversation_context}"
                f"CONTEXT (use ONLY this information):\n{context}\n\n"
                f"User Query: {q}\n\n"
                f"INSTRUCTIONS:\n"
                f"- Provide ONLY a brief conversational response\n"
                f"- Do NOT list or format property details\n"
                f"- If properties found, just say how many match the criteria\n"
                f"- If no matches, say 'No matching properties found'\n"
                f"- Answer the current query directly - do NOT question the user's request\n"
                f"- Keep response under 2 sentences\n\n"
                "Brief Response:"
            )
            
            answer_text = ""
            if args.llm_engine == "llama" and args.llama_model_path:
                answer_text = call_llm_llamacpp(prompt, args.llama_model_path)
            elif args.llm_engine == "gpt4all":
                answer_text = call_llm_gpt4all(prompt, args.gpt4all_model)
            elif args.llm_engine == "ollama":
                answer_text = call_llm_ollama(prompt, args.ollama_model)
            elif args.llm_engine == "groq" and args.groq_api_key:
                answer_text = call_llm_groq(prompt, args.groq_model, args.groq_api_key)
            elif args.llm_engine == "cohere" and args.cohere_api_key:
                answer_text = call_llm_cohere(prompt, args.cohere_model, args.cohere_api_key)
            else:
                answer_text = f"Found {len(listings)} matching properties" if listings else "No matching properties found"
            
            dates = [x.get("scraped_at") for x in listings if x.get("scraped_at")]
            if dates:
                freshness = f"Data scraped between {min(dates)} and {max(dates)}."
            else:
                freshness = "Data freshness unknown (scraped_at not available)."
            
            # Ensure we only show the exact number requested
            final_listings = listings[:actual_top_n] if listings else []
            
            # Validate response quality
            quality_report = validate_response_quality(q, answer_text, final_listings)
            
            # Add quality suggestions if score is low
            formatted_response = format_answer(q, final_listings, freshness, answer_text)
            
            # Check for recommendation requests
            if analysis['intent'] == 'recommend' and final_listings:
                # Get similar properties for the first result
                similar_props = get_similar_properties(final_listings[0]['listing_id'], processed_map, 2)
                if similar_props:
                    formatted_response += "\n\nSIMILAR PROPERTIES YOU MIGHT LIKE:\n"
                    formatted_response += "=" * 40 + "\n"
                    for i, prop in enumerate(similar_props, 1):
                        formatted_response += f"\n{i}. {prop.get('title', 'No title')}\n"
                        formatted_response += f"   Price: PKR {prop.get('price_numeric', 0):,}\n"
                        if prop.get('bedrooms'):
                            formatted_response += f"   Bedrooms: {prop['bedrooms']}\n"
            
            if quality_report['overall'] < 0.7 and quality_report['suggestions']:
                formatted_response += f"\n\nSuggestions: {'; '.join(quality_report['suggestions'])}"
            
            return formatted_response
        
        else:
            # No relevant property data found - forward to LLM for general conversation
            if args.explain:
                print("No relevant property data found. Forwarding to LLM for general conversation.\n")

            # Include conversation history in general chat
            conversation_context = ""
            if conversation_history:
                conversation_context = "Previous conversation:\n"
                for exchange in conversation_history[-3:]:  # Last 3 exchanges
                    conversation_context += f"User: {exchange['user']}\n"
                    conversation_context += f"Assistant: {exchange['assistant']}\n\n"
                conversation_context += "Current query:\n"

            general_prompt = (
                f"{conversation_context}You are a helpful AI assistant specializing in real estate. "
                f"The user asked: '{q}'. Please provide a helpful and friendly response considering the conversation context."
            )
            answer_text = call_llm_for_general_chat(general_prompt)
            return answer_text
            print("-" * 60)

    if args.interactive:
        print("=" * 60)
        print("REAL ESTATE RAG ASSISTANT - Interactive Chat")
        print("=" * 60)
        print("Ask me about properties in Bahria Town Phase 7!")
        print("Examples:")
        print("  - Show me 10 marla houses under 5 crore")
        print("  - Find apartments with 2 bedrooms")
        print("  - What's available near parks?")
        print("\nType 'exit', 'quit', or press Ctrl+C to quit.")
        print("-" * 60)
        
        # Initialize conversation history
        conversation_history = []
        
        try:
            while True:
                print()  # Add some spacing
                q = input("You: ").strip()
                if not q:
                    continue
                if q.lower() in {"exit", "quit", "bye", "goodbye"}:
                    print("\nGoodbye! Happy house hunting!")
                    break
                
                print("Assistant: Searching...")
                response = answer_one(q, conversation_history)
                print(f"\n{response}")
                
                # Add to conversation history
                conversation_history.append({"user": q, "assistant": response})
                
                print("-" * 60)
                
        except KeyboardInterrupt:
            print("\n\nGoodbye! Happy house hunting!")
        return 0
    else:
        if not args.query:
            print("--query is required unless --interactive is used")
            return 1
        answer_one(args.query)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

