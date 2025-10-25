"""
Challenge Submission and Result Storage Module
Handles API submissions, audio storage, and performance tracking
"""
import json
import time
import shutil
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
import requests
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ChallengeResultStorage:
    """Stores challenge results, audio samples, and tracks performance"""
    
    def __init__(self, storage_dir: str = "challenge_results"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.audio_dir = self.storage_dir / "audio_samples"
        self.audio_dir.mkdir(exist_ok=True)
        
        self.results_file = self.storage_dir / "results.jsonl"
        self.stats_file = self.storage_dir / "statistics.json"
        
        # Load existing stats
        self.stats = self._load_stats()
    
    def _load_stats(self) -> Dict:
        """Load statistics from file or create new"""
        if self.stats_file.exists():
            with open(self.stats_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'total_attempts': 0,
                'correct': 0,
                'wrong': 0,
                'total_score': 0,
                'accuracy': 0.0,
                'predictions': {},  # class -> count
                'confusion_matrix': {},  # predicted -> actual -> count
                'start_time': datetime.now().isoformat(),
                'last_update': datetime.now().isoformat()
            }
    
    def _save_stats(self):
        """Save statistics to file"""
        self.stats['last_update'] = datetime.now().isoformat()
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def store_result(
        self,
        challenge_id: str,
        audio_path: str,
        prediction: str,
        confidence: float,
        all_probabilities: Dict[str, float],
        is_correct: bool,
        score_awarded: int,
        actual_label: Optional[str] = None,
        inference_time: float = 0.0,
        response_data: Optional[Dict] = None
    ):
        """
        Store a challenge result
        
        Args:
            challenge_id: Challenge UUID
            audio_path: Path to audio file
            prediction: Predicted class
            confidence: Prediction confidence
            all_probabilities: All class probabilities
            is_correct: Whether prediction was correct
            score_awarded: Points awarded
            actual_label: True label (if known)
            inference_time: Time taken for inference
            response_data: Full API response
        """
        timestamp = datetime.now().isoformat()
        
        # Store audio file with metadata
        audio_filename = f"{challenge_id}_{prediction}_{timestamp.replace(':', '-')}.wav"
        stored_audio_path = self.audio_dir / audio_filename
        
        if Path(audio_path).exists():
            shutil.copy(audio_path, stored_audio_path)
        
        # Create result entry
        result = {
            'timestamp': timestamp,
            'challenge_id': challenge_id,
            'prediction': prediction,
            'confidence': confidence,
            'all_probabilities': all_probabilities,
            'is_correct': is_correct,
            'score_awarded': score_awarded,
            'actual_label': actual_label,
            'inference_time': inference_time,
            'audio_file': str(stored_audio_path),
            'response_data': response_data
        }
        
        # Append to results file
        with open(self.results_file, 'a') as f:
            f.write(json.dumps(result) + '\n')
        
        # Update statistics
        self.stats['total_attempts'] += 1
        if is_correct:
            self.stats['correct'] += 1
        else:
            self.stats['wrong'] += 1
        
        self.stats['total_score'] += score_awarded
        self.stats['accuracy'] = self.stats['correct'] / self.stats['total_attempts']
        
        # Track predictions
        if prediction not in self.stats['predictions']:
            self.stats['predictions'][prediction] = 0
        self.stats['predictions'][prediction] += 1
        
        # Update confusion matrix if we know the actual label
        if actual_label:
            if prediction not in self.stats['confusion_matrix']:
                self.stats['confusion_matrix'][prediction] = {}
            if actual_label not in self.stats['confusion_matrix'][prediction]:
                self.stats['confusion_matrix'][prediction][actual_label] = 0
            self.stats['confusion_matrix'][prediction][actual_label] += 1
        
        self._save_stats()
    
    def get_statistics(self) -> Dict:
        """Get current statistics"""
        return self.stats.copy()
    
    def get_recent_results(self, n: int = 10) -> List[Dict]:
        """Get n most recent results"""
        if not self.results_file.exists():
            return []
        
        results = []
        with open(self.results_file, 'r') as f:
            for line in f:
                results.append(json.loads(line))
        
        return results[-n:]
    
    def analyze_performance(self) -> Dict:
        """Analyze performance and return insights"""
        if self.stats['total_attempts'] == 0:
            return {'message': 'No attempts yet'}
        
        analysis = {
            'overall_accuracy': self.stats['accuracy'],
            'total_score': self.stats['total_score'],
            'avg_score_per_attempt': self.stats['total_score'] / self.stats['total_attempts'],
            'most_predicted_class': max(self.stats['predictions'], key=self.stats['predictions'].get) if self.stats['predictions'] else None,
            'prediction_distribution': self.stats['predictions']
        }
        
        # Analyze recent performance
        recent = self.get_recent_results(n=10)
        if recent:
            recent_correct = sum(1 for r in recent if r['is_correct'])
            analysis['recent_accuracy'] = recent_correct / len(recent)
            analysis['recent_avg_confidence'] = np.mean([r['confidence'] for r in recent])
        
        return analysis
    
    def export_for_training(self, output_dir: str):
        """Export stored audio samples organized by prediction for retraining"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Read all results
        results = []
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                for line in f:
                    results.append(json.loads(line))
        
        # Organize by prediction
        for result in results:
            pred_class = result['prediction']
            class_dir = output_path / pred_class
            class_dir.mkdir(exist_ok=True)
            
            # Copy audio file
            audio_file = Path(result['audio_file'])
            if audio_file.exists():
                dest = class_dir / audio_file.name
                shutil.copy(audio_file, dest)
        
        print(f"Exported {len(results)} samples to {output_dir}")


class ChallengeAPIClient:
    """Handle API communication with challenge server"""
    
    def __init__(
        self,
        api_base_url: str = None,
        api_token: str = None
    ):
        """
        Initialize API client
        
        Args:
            api_base_url: API base URL (defaults to env var or https://edth.helsing.codes)
            api_token: API token (defaults to env var)
        """
        # Load from environment variables if not provided
        if api_base_url is None:
            api_base_url = os.getenv('API_BASE_URL', 'https://edth.helsing.codes')
        
        if api_token is None:
            api_token = os.getenv('API_TOKEN')
            if api_token is None:
                raise ValueError(
                    "API_TOKEN not found! Please set it in .env file.\n"
                    "Create a .env file with: API_TOKEN=your_token_here"
                )
        
        self.api_base_url = api_base_url
        self.api_token = api_token
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json'
        })
    
    def get_challenge(self) -> Dict:
        """Fetch current challenge"""
        response = self.session.get(
            f"{self.api_base_url}/api/challenge",
            timeout=100  # Increased to 100s to handle very slow server responses
        )
        response.raise_for_status()
        return response.json()
    
    def submit_classification(self, challenge_id: str, classification: str) -> Dict:
        """Submit classification for a challenge"""
        payload = {
            'challenge_id': challenge_id,
            'classification': classification
        }
        
        response = self.session.post(
            f"{self.api_base_url}/api/challenge",
            json=payload,
            timeout=100  # Increased to 100s to handle very slow server responses
        )
        response.raise_for_status()
        return response.json()
    
    def download_audio(self, wav_url: str, output_path: str) -> bool:
        """Download audio file from challenge"""
        full_url = f"{self.api_base_url}{wav_url}"
        response = self.session.get(full_url, timeout=100)  # Increased to 100s
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        return True


class AdaptiveLearningTracker:
    """Track predictions to identify patterns and adapt strategy"""
    
    def __init__(self):
        self.history = []
        self.class_performance = {}
    
    def add_result(self, prediction: str, confidence: float, is_correct: bool):
        """Add a prediction result"""
        self.history.append({
            'prediction': prediction,
            'confidence': confidence,
            'is_correct': is_correct,
            'timestamp': time.time()
        })
        
        # Track per-class performance
        if prediction not in self.class_performance:
            self.class_performance[prediction] = {
                'attempts': 0,
                'correct': 0,
                'avg_confidence': 0.0
            }
        
        stats = self.class_performance[prediction]
        stats['attempts'] += 1
        if is_correct:
            stats['correct'] += 1
        
        # Update rolling average confidence
        stats['avg_confidence'] = (
            stats['avg_confidence'] * (stats['attempts'] - 1) + confidence
        ) / stats['attempts']
    
    def should_adjust_threshold(self, prediction: str, confidence: float) -> bool:
        """Determine if we should adjust confidence threshold for a class"""
        if prediction not in self.class_performance:
            return False
        
        stats = self.class_performance[prediction]
        if stats['attempts'] < 5:  # Need enough samples
            return False
        
        # If accuracy is low and confidence is high, we're overconfident
        accuracy = stats['correct'] / stats['attempts']
        if accuracy < 0.6 and confidence > 0.8:
            return True
        
        return False
    
    def get_recommendations(self) -> Dict:
        """Get recommendations based on performance"""
        if len(self.history) < 10:
            return {'message': 'Not enough data for recommendations'}
        
        recommendations = {}
        
        for class_name, stats in self.class_performance.items():
            if stats['attempts'] < 3:
                continue
            
            accuracy = stats['correct'] / stats['attempts']
            
            if accuracy < 0.5:
                recommendations[class_name] = {
                    'status': 'poor',
                    'suggestion': 'Consider reviewing training data for this class'
                }
            elif accuracy < 0.7:
                recommendations[class_name] = {
                    'status': 'moderate',
                    'suggestion': 'May need more training examples'
                }
            else:
                recommendations[class_name] = {
                    'status': 'good',
                    'suggestion': 'Continue with current approach'
                }
        
        return recommendations
