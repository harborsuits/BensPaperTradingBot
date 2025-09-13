"""
Rating and review system for the marketplace API.

Provides:
- Component rating and statistics
- User reviews and feedback
- Reputation scores
- Moderation capabilities
"""

import os
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple

# Configure logging
logger = logging.getLogger(__name__)

class RatingSystem:
    """
    Manages ratings for marketplace components.
    
    Provides functionality for:
    - Adding and updating ratings
    - Calculating average ratings
    - Rating analytics
    """
    
    def __init__(self, ratings_path: Optional[str] = None):
        """
        Initialize the rating system.
        
        Args:
            ratings_path: Optional custom path for ratings data
        """
        self.ratings_path = ratings_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "marketplace", "ratings"
        )
        
        # Create ratings directory
        os.makedirs(self.ratings_path, exist_ok=True)
        
        logger.info(f"Rating system initialized with path: {self.ratings_path}")
    
    def add_rating(self, component_id: str, rating: float, user_id: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a rating for a component.
        
        Args:
            component_id: ID of the component
            rating: Rating value (1-5)
            user_id: ID of the user submitting the rating
            version: Specific version being rated (None for latest)
        
        Returns:
            Dict with updated rating information
        """
        if rating < 1 or rating > 5:
            raise ValueError("Rating must be between 1 and 5")
        
        # Get component ratings
        ratings = self._get_component_ratings(component_id, version)
        
        # Add or update user rating
        ratings["ratings"][user_id] = {
            "value": rating,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update averages
        self._calculate_average(ratings)
        
        # Save ratings
        self._save_component_ratings(component_id, ratings, version)
        
        logger.info(f"Added rating {rating} for component {component_id}" + 
                   (f" version {version}" if version else ""))
        
        return {
            "component_id": component_id,
            "version": version or "latest",
            "average_rating": ratings["average_rating"],
            "ratings_count": ratings["ratings_count"],
            "rating_distribution": ratings["rating_distribution"]
        }
    
    def get_ratings(self, component_id: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get ratings for a component.
        
        Args:
            component_id: ID of the component
            version: Specific version (None for latest)
        
        Returns:
            Dict with rating information
        """
        ratings = self._get_component_ratings(component_id, version)
        
        # Return public data (exclude individual user ratings)
        return {
            "component_id": component_id,
            "version": version or "latest",
            "average_rating": ratings["average_rating"],
            "ratings_count": ratings["ratings_count"],
            "rating_distribution": ratings["rating_distribution"]
        }
    
    def get_user_rating(self, component_id: str, user_id: str, version: Optional[str] = None) -> Optional[float]:
        """
        Get a user's rating for a component.
        
        Args:
            component_id: ID of the component
            user_id: ID of the user
            version: Specific version (None for latest)
        
        Returns:
            User's rating value, or None if not rated
        """
        ratings = self._get_component_ratings(component_id, version)
        
        # Get user rating
        user_rating = ratings["ratings"].get(user_id)
        
        if user_rating:
            return user_rating["value"]
        else:
            return None
    
    def delete_rating(self, component_id: str, user_id: str, version: Optional[str] = None) -> bool:
        """
        Delete a user's rating for a component.
        
        Args:
            component_id: ID of the component
            user_id: ID of the user
            version: Specific version (None for latest)
        
        Returns:
            True if successful, False otherwise
        """
        # Get component ratings
        ratings = self._get_component_ratings(component_id, version)
        
        # Check if user has rated
        if user_id not in ratings["ratings"]:
            return False
        
        # Remove rating
        del ratings["ratings"][user_id]
        
        # Update averages
        self._calculate_average(ratings)
        
        # Save ratings
        self._save_component_ratings(component_id, ratings, version)
        
        logger.info(f"Deleted rating by user {user_id} for component {component_id}" + 
                   (f" version {version}" if version else ""))
        
        return True
    
    def _get_component_ratings(self, component_id: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get ratings data for a component.
        
        Args:
            component_id: ID of the component
            version: Specific version (None for latest)
        
        Returns:
            Dict with ratings data
        """
        version_str = version or "latest"
        ratings_file = os.path.join(self.ratings_path, f"{component_id}_{version_str}_ratings.json")
        
        if os.path.exists(ratings_file):
            try:
                with open(ratings_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Invalid ratings file for {component_id} {version_str}")
        
        # Return default empty ratings
        return {
            "component_id": component_id,
            "version": version_str,
            "average_rating": 0.0,
            "ratings_count": 0,
            "rating_distribution": {
                "1": 0, "2": 0, "3": 0, "4": 0, "5": 0
            },
            "ratings": {}
        }
    
    def _save_component_ratings(self, component_id: str, ratings: Dict[str, Any], version: Optional[str] = None) -> None:
        """
        Save ratings data for a component.
        
        Args:
            component_id: ID of the component
            ratings: Ratings data
            version: Specific version (None for latest)
        """
        version_str = version or "latest"
        ratings_file = os.path.join(self.ratings_path, f"{component_id}_{version_str}_ratings.json")
        
        try:
            with open(ratings_file, 'w') as f:
                json.dump(ratings, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save ratings for {component_id} {version_str}: {e}")
    
    def _calculate_average(self, ratings: Dict[str, Any]) -> None:
        """
        Calculate average rating and distribution.
        
        Args:
            ratings: Ratings data to update
        """
        user_ratings = ratings["ratings"]
        
        if not user_ratings:
            ratings["average_rating"] = 0.0
            ratings["ratings_count"] = 0
            ratings["rating_distribution"] = {
                "1": 0, "2": 0, "3": 0, "4": 0, "5": 0
            }
            return
        
        # Count ratings
        ratings_count = len(user_ratings)
        
        # Calculate average
        total = sum(r["value"] for r in user_ratings.values())
        average = total / ratings_count
        
        # Calculate distribution
        distribution = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
        for r in user_ratings.values():
            key = str(int(r["value"]))
            distribution[key] += 1
        
        # Update ratings data
        ratings["average_rating"] = average
        ratings["ratings_count"] = ratings_count
        ratings["rating_distribution"] = distribution


class ReviewManager:
    """
    Manages reviews for marketplace components.
    
    Provides functionality for:
    - Adding and updating reviews
    - Review moderation
    - Review analytics
    """
    
    def __init__(self, reviews_path: Optional[str] = None):
        """
        Initialize the review manager.
        
        Args:
            reviews_path: Optional custom path for reviews data
        """
        self.reviews_path = reviews_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "marketplace", "reviews"
        )
        
        # Create reviews directory
        os.makedirs(self.reviews_path, exist_ok=True)
        
        logger.info(f"Review manager initialized with path: {self.reviews_path}")
    
    def add_review(self, component_id: str, review_id: str, review_text: str, user_id: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a review for a component.
        
        Args:
            component_id: ID of the component
            review_id: ID of the review
            review_text: Text content of the review
            user_id: ID of the user submitting the review
            version: Specific version being reviewed (None for latest)
        
        Returns:
            Dict with review information
        """
        # Get component reviews
        reviews = self._get_component_reviews(component_id, version)
        
        # Create review object
        review = {
            "review_id": review_id,
            "user_id": user_id,
            "text": review_text,
            "timestamp": datetime.now().isoformat(),
            "status": "active",
            "helpful_count": 0,
            "reported": False
        }
        
        # Add review
        reviews["reviews"][review_id] = review
        reviews["review_count"] = len(reviews["reviews"])
        
        # Save reviews
        self._save_component_reviews(component_id, reviews, version)
        
        logger.info(f"Added review for component {component_id}" + 
                   (f" version {version}" if version else ""))
        
        return review
    
    def get_reviews(self, component_id: str, version: Optional[str] = None, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get reviews for a component.
        
        Args:
            component_id: ID of the component
            version: Specific version (None for latest)
            limit: Maximum number of reviews to return
            offset: Offset for pagination
        
        Returns:
            List of review objects
        """
        reviews = self._get_component_reviews(component_id, version)
        
        # Get active reviews
        active_reviews = [
            r for r in reviews["reviews"].values()
            if r["status"] == "active"
        ]
        
        # Sort by date (newest first)
        sorted_reviews = sorted(
            active_reviews, 
            key=lambda r: r["timestamp"],
            reverse=True
        )
        
        # Apply pagination
        paginated = sorted_reviews[offset:offset + limit]
        
        return paginated
    
    def update_review(self, component_id: str, review_id: str, review_text: str, user_id: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Update a review for a component.
        
        Args:
            component_id: ID of the component
            review_id: ID of the review
            review_text: New text content of the review
            user_id: ID of the user updating the review
            version: Specific version (None for latest)
        
        Returns:
            Updated review object, or None if not found
        """
        # Get component reviews
        reviews = self._get_component_reviews(component_id, version)
        
        # Check if review exists
        if review_id not in reviews["reviews"]:
            return None
        
        # Check if user owns the review
        review = reviews["reviews"][review_id]
        if review["user_id"] != user_id:
            return None
        
        # Update review
        review["text"] = review_text
        review["timestamp"] = datetime.now().isoformat()
        review["edited"] = True
        
        # Save reviews
        self._save_component_reviews(component_id, reviews, version)
        
        logger.info(f"Updated review {review_id} for component {component_id}" + 
                   (f" version {version}" if version else ""))
        
        return review
    
    def delete_review(self, component_id: str, review_id: str, user_id: str, version: Optional[str] = None) -> bool:
        """
        Delete a review for a component.
        
        Args:
            component_id: ID of the component
            review_id: ID of the review
            user_id: ID of the user deleting the review
            version: Specific version (None for latest)
        
        Returns:
            True if successful, False otherwise
        """
        # Get component reviews
        reviews = self._get_component_reviews(component_id, version)
        
        # Check if review exists
        if review_id not in reviews["reviews"]:
            return False
        
        # Check if user owns the review
        review = reviews["reviews"][review_id]
        if review["user_id"] != user_id:
            return False
        
        # Remove review
        del reviews["reviews"][review_id]
        reviews["review_count"] = len(reviews["reviews"])
        
        # Save reviews
        self._save_component_reviews(component_id, reviews, version)
        
        logger.info(f"Deleted review {review_id} for component {component_id}" + 
                   (f" version {version}" if version else ""))
        
        return True
    
    def report_review(self, component_id: str, review_id: str, reason: str, user_id: str, version: Optional[str] = None) -> bool:
        """
        Report a review for moderation.
        
        Args:
            component_id: ID of the component
            review_id: ID of the review
            reason: Reason for reporting
            user_id: ID of the user reporting the review
            version: Specific version (None for latest)
        
        Returns:
            True if successful, False otherwise
        """
        # Get component reviews
        reviews = self._get_component_reviews(component_id, version)
        
        # Check if review exists
        if review_id not in reviews["reviews"]:
            return False
        
        # Mark as reported
        review = reviews["reviews"][review_id]
        review["reported"] = True
        
        if "reports" not in review:
            review["reports"] = []
        
        # Add report
        review["reports"].append({
            "user_id": user_id,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })
        
        # Save reviews
        self._save_component_reviews(component_id, reviews, version)
        
        logger.info(f"Reported review {review_id} for component {component_id}" + 
                   (f" version {version}" if version else ""))
        
        return True
    
    def moderate_review(self, component_id: str, review_id: str, action: str, admin_id: str, version: Optional[str] = None) -> bool:
        """
        Moderate a review (admin function).
        
        Args:
            component_id: ID of the component
            review_id: ID of the review
            action: Moderation action ('approve', 'hide', 'delete')
            admin_id: ID of the admin performing the action
            version: Specific version (None for latest)
        
        Returns:
            True if successful, False otherwise
        """
        # Get component reviews
        reviews = self._get_component_reviews(component_id, version)
        
        # Check if review exists
        if review_id not in reviews["reviews"]:
            return False
        
        # Apply moderation action
        review = reviews["reviews"][review_id]
        
        if action == "approve":
            review["status"] = "active"
            review["reported"] = False
            if "reports" in review:
                del review["reports"]
        elif action == "hide":
            review["status"] = "hidden"
        elif action == "delete":
            del reviews["reviews"][review_id]
            reviews["review_count"] = len(reviews["reviews"])
        else:
            return False
        
        # Save reviews
        self._save_component_reviews(component_id, reviews, version)
        
        logger.info(f"Moderated review {review_id} with action {action} for component {component_id}" + 
                   (f" version {version}" if version else ""))
        
        return True
    
    def mark_helpful(self, component_id: str, review_id: str, user_id: str, version: Optional[str] = None) -> bool:
        """
        Mark a review as helpful.
        
        Args:
            component_id: ID of the component
            review_id: ID of the review
            user_id: ID of the user marking the review
            version: Specific version (None for latest)
        
        Returns:
            True if successful, False otherwise
        """
        # Get component reviews
        reviews = self._get_component_reviews(component_id, version)
        
        # Check if review exists
        if review_id not in reviews["reviews"]:
            return False
        
        # Update helpful count
        review = reviews["reviews"][review_id]
        
        if "helpful_users" not in review:
            review["helpful_users"] = []
        
        # Check if user already marked as helpful
        if user_id in review["helpful_users"]:
            return False
        
        # Add user and increment count
        review["helpful_users"].append(user_id)
        review["helpful_count"] = len(review["helpful_users"])
        
        # Save reviews
        self._save_component_reviews(component_id, reviews, version)
        
        logger.info(f"Marked review {review_id} as helpful for component {component_id}" + 
                   (f" version {version}" if version else ""))
        
        return True
    
    def _get_component_reviews(self, component_id: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get reviews data for a component.
        
        Args:
            component_id: ID of the component
            version: Specific version (None for latest)
        
        Returns:
            Dict with reviews data
        """
        version_str = version or "latest"
        reviews_file = os.path.join(self.reviews_path, f"{component_id}_{version_str}_reviews.json")
        
        if os.path.exists(reviews_file):
            try:
                with open(reviews_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Invalid reviews file for {component_id} {version_str}")
        
        # Return default empty reviews
        return {
            "component_id": component_id,
            "version": version_str,
            "review_count": 0,
            "reviews": {}
        }
    
    def _save_component_reviews(self, component_id: str, reviews: Dict[str, Any], version: Optional[str] = None) -> None:
        """
        Save reviews data for a component.
        
        Args:
            component_id: ID of the component
            reviews: Reviews data
            version: Specific version (None for latest)
        """
        version_str = version or "latest"
        reviews_file = os.path.join(self.reviews_path, f"{component_id}_{version_str}_reviews.json")
        
        try:
            with open(reviews_file, 'w') as f:
                json.dump(reviews, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save reviews for {component_id} {version_str}: {e}")


class CommunityMetrics:
    """
    Manages community metrics for marketplace components.
    
    Provides analytics on:
    - User contributions
    - Component popularity
    - Community activity
    """
    
    def __init__(self, metrics_path: Optional[str] = None):
        """
        Initialize the community metrics.
        
        Args:
            metrics_path: Optional custom path for metrics data
        """
        self.metrics_path = metrics_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "marketplace", "metrics"
        )
        
        # Create metrics directory
        os.makedirs(self.metrics_path, exist_ok=True)
        
        logger.info(f"Community metrics initialized with path: {self.metrics_path}")
    
    def track_component_view(self, component_id: str, user_id: Optional[str] = None, version: Optional[str] = None) -> None:
        """
        Track a component view.
        
        Args:
            component_id: ID of the component
            user_id: Optional ID of the viewing user
            version: Specific version (None for latest)
        """
        # Get component metrics
        metrics = self._get_component_metrics(component_id, version)
        
        # Update view count
        metrics["view_count"] += 1
        
        # Update last viewed
        metrics["last_viewed"] = datetime.now().isoformat()
        
        # Track unique viewers
        if user_id:
            if "viewers" not in metrics:
                metrics["viewers"] = []
            
            if user_id not in metrics["viewers"]:
                metrics["viewers"].append(user_id)
                metrics["unique_viewers"] = len(metrics["viewers"])
        
        # Save metrics
        self._save_component_metrics(component_id, metrics, version)
    
    def track_component_download(self, component_id: str, user_id: Optional[str] = None, version: Optional[str] = None) -> None:
        """
        Track a component download.
        
        Args:
            component_id: ID of the component
            user_id: Optional ID of the downloading user
            version: Specific version (None for latest)
        """
        # Get component metrics
        metrics = self._get_component_metrics(component_id, version)
        
        # Update download count
        metrics["download_count"] += 1
        
        # Update last downloaded
        metrics["last_downloaded"] = datetime.now().isoformat()
        
        # Track unique downloaders
        if user_id:
            if "downloaders" not in metrics:
                metrics["downloaders"] = []
            
            if user_id not in metrics["downloaders"]:
                metrics["downloaders"].append(user_id)
                metrics["unique_downloaders"] = len(metrics["downloaders"])
        
        # Save metrics
        self._save_component_metrics(component_id, metrics, version)
    
    def track_component_usage(self, component_id: str, user_id: str, version: Optional[str] = None) -> None:
        """
        Track component usage.
        
        Args:
            component_id: ID of the component
            user_id: ID of the user using the component
            version: Specific version (None for latest)
        """
        # Get component metrics
        metrics = self._get_component_metrics(component_id, version)
        
        # Update usage count
        metrics["usage_count"] += 1
        
        # Update last used
        metrics["last_used"] = datetime.now().isoformat()
        
        # Track unique users
        if "users" not in metrics:
            metrics["users"] = []
        
        if user_id not in metrics["users"]:
            metrics["users"].append(user_id)
            metrics["unique_users"] = len(metrics["users"])
        
        # Save metrics
        self._save_component_metrics(component_id, metrics, version)
    
    def get_component_metrics(self, component_id: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metrics for a component.
        
        Args:
            component_id: ID of the component
            version: Specific version (None for latest)
        
        Returns:
            Dict with metrics data
        """
        metrics = self._get_component_metrics(component_id, version)
        
        # Return public data (exclude user IDs)
        return {
            "component_id": metrics["component_id"],
            "version": metrics["version"],
            "view_count": metrics["view_count"],
            "download_count": metrics["download_count"],
            "usage_count": metrics["usage_count"],
            "unique_viewers": metrics.get("unique_viewers", 0),
            "unique_downloaders": metrics.get("unique_downloaders", 0),
            "unique_users": metrics.get("unique_users", 0),
            "last_viewed": metrics.get("last_viewed"),
            "last_downloaded": metrics.get("last_downloaded"),
            "last_used": metrics.get("last_used")
        }
    
    def get_popular_components(self, limit: int = 10, metric: str = "download_count") -> List[Dict[str, Any]]:
        """
        Get popular components based on metrics.
        
        Args:
            limit: Maximum number of components to return
            metric: Metric to sort by ('download_count', 'view_count', 'usage_count')
        
        Returns:
            List of component metrics
        """
        # Get all metrics files
        metrics_files = [
            f for f in os.listdir(self.metrics_path)
            if f.endswith("_metrics.json")
        ]
        
        # Load metrics
        components = []
        for file in metrics_files:
            metrics_file = os.path.join(self.metrics_path, file)
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    
                    # Add public data
                    components.append({
                        "component_id": metrics["component_id"],
                        "version": metrics["version"],
                        "view_count": metrics["view_count"],
                        "download_count": metrics["download_count"],
                        "usage_count": metrics["usage_count"],
                        "unique_viewers": metrics.get("unique_viewers", 0),
                        "unique_downloaders": metrics.get("unique_downloaders", 0),
                        "unique_users": metrics.get("unique_users", 0)
                    })
            except:
                continue
        
        # Sort by metric
        if metric not in ["download_count", "view_count", "usage_count"]:
            metric = "download_count"
        
        sorted_components = sorted(
            components,
            key=lambda c: c.get(metric, 0),
            reverse=True
        )
        
        # Apply limit
        return sorted_components[:limit]
    
    def _get_component_metrics(self, component_id: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metrics data for a component.
        
        Args:
            component_id: ID of the component
            version: Specific version (None for latest)
        
        Returns:
            Dict with metrics data
        """
        version_str = version or "latest"
        metrics_file = os.path.join(self.metrics_path, f"{component_id}_{version_str}_metrics.json")
        
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Invalid metrics file for {component_id} {version_str}")
        
        # Return default empty metrics
        return {
            "component_id": component_id,
            "version": version_str,
            "view_count": 0,
            "download_count": 0,
            "usage_count": 0
        }
    
    def _save_component_metrics(self, component_id: str, metrics: Dict[str, Any], version: Optional[str] = None) -> None:
        """
        Save metrics data for a component.
        
        Args:
            component_id: ID of the component
            metrics: Metrics data
            version: Specific version (None for latest)
        """
        version_str = version or "latest"
        metrics_file = os.path.join(self.metrics_path, f"{component_id}_{version_str}_metrics.json")
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metrics for {component_id} {version_str}: {e}")
