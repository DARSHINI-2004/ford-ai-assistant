# recommend/recommender.py
"""
Rule-based vehicle recommender.

Design:
- Map common user intents to vehicle attributes.
- Score vehicles by matching attributes and simple heuristics.
- Return top N vehicles with explanation.

Examples:
- "family SUV" -> Ford Explorer, Ford Escape
- "pickup for towing" -> Ford F-150, Ford Ranger
"""

from typing import List, Dict
import json
import os
import re

DATA_FILE = os.path.join("data", "ford_vehicles.json")


class Recommender:
    def __init__(self, data_file=DATA_FILE):
        with open(data_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)["vehicles"]

    def _score_vehicle(self, vehicle: Dict, intent: str) -> int:
        """
        Simple scoring:
        - +3 if type matches (SUV, Pickup, Coupe)
        - +2 if seating capacity >= requested (family)
        - +2 if towing/payload related keywords and vehicle is pickup
        - +1 for safety features match if 'safe' or 'family' in intent
        """
        score = 0
        intent_lower = intent.lower()

        # Type matching
        if "suv" in intent_lower and vehicle["type"].lower() == "suv":
            score += 3
        if "compact" in intent_lower and "compact" in vehicle["type"].lower():
            score += 2
        if "pickup" in intent_lower or "towing" in intent_lower or "tow" in intent_lower:
            if vehicle["type"].lower() == "pickup":
                score += 3

        # Family intent: prefer seating >= 5 and safety features
        if "family" in intent_lower:
            if vehicle.get("seating_capacity", 0) >= 5:
                score += 2
            # safety features
            if any("lane" in s.lower() or "adaptive" in s.lower() or "blind" in s.lower() for s in vehicle.get("safety_features", [])):
                score += 1

        # Economy / commute
        if "commute" in intent_lower or "economy" in intent_lower or "fuel" in intent_lower:
            if "hybrid" in vehicle.get("fuel_type", "").lower():
                score += 2
            if "escape" in vehicle["model"].lower():
                score += 1

        # Sport / performance
        if "sport" in intent_lower or "performance" in intent_lower or "fast" in intent_lower:
            if "mustang" in vehicle["model"].lower():
                score += 3

        # Tiebreaker: prefer higher seating for family, prefer pickups for towing
        return score

    def recommend(self, query: str, top_n: int = 2) -> List[Dict]:
        scored = []
        for v in self.data:
            s = self._score_vehicle(v, query)
            scored.append((s, v))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [v for score, v in scored[:top_n] if score > 0]
        explanations = []
        for score, v in scored[:top_n]:
            if score <= 0:
                continue
            explanation = self._explain_score(v, query)
            explanations.append({"model": v["model"], "score": score, "explanation": explanation})
        # If no vehicle matched, provide a fallback: top 2 by default
        if not explanations:
            fallback = [{"model": v["model"], "score": 0, "explanation": "No strong match; default suggestion."} for v in [s[1] for s in scored[:top_n]]]
            return fallback
        return explanations

    def _explain_score(self, vehicle: Dict, intent: str) -> str:
        intent_lower = intent.lower()
        reasons = []
        if "suv" in intent_lower and vehicle["type"].lower() == "suv":
            reasons.append("Matches requested SUV type")
        if "pickup" in intent_lower and vehicle["type"].lower() == "pickup":
            reasons.append("Pickup suitable for towing/payload")
        if "family" in intent_lower and vehicle.get("seating_capacity", 0) >= 5:
            reasons.append(f"Seating capacity {vehicle.get('seating_capacity')} suitable for family")
        if "fuel" in intent_lower and "hybrid" in vehicle.get("fuel_type", "").lower():
            reasons.append("Hybrid option for better fuel economy")
        if not reasons:
            reasons.append("General purpose match based on vehicle attributes")
        return "; ".join(reasons)
