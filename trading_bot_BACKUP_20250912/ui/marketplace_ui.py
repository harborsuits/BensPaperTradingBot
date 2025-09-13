"""
Marketplace UI for the trading platform.

Provides a user interface for:
- Browsing and searching components
- Publishing components to the marketplace
- Managing user components
- Rating and reviewing components
- Version management and dependency resolution
"""

import os
import json
import uuid
import base64
import logging
import tempfile
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union

from trading_bot.marketplace.api.marketplace_api import MarketplaceAPI
from trading_bot.marketplace.api.security import SecurityManager
from trading_bot.marketplace.api.version_manager import VersionManager

# Configure logging
logger = logging.getLogger(__name__)

class MarketplaceUI:
    """
    User interface for the component marketplace.
    
    Provides:
    - Component browsing and search
    - Publishing interface
    - Component management
    - Rating and review interface
    """
    
    def __init__(self, marketplace_api: Optional[MarketplaceAPI] = None):
        """
        Initialize the marketplace UI.
        
        Args:
            marketplace_api: Optional marketplace API instance
        """
        # Initialize marketplace API if not provided
        if marketplace_api is None:
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            marketplace_path = os.path.join(base_path, "marketplace")
            self.api = MarketplaceAPI(marketplace_path=marketplace_path)
        else:
            self.api = marketplace_api
        
        # Component types and icons
        self.component_types = {
            "SIGNAL_GENERATOR": "📊",
            "FILTER": "🔍",
            "POSITION_SIZER": "📏",
            "EXIT_MANAGER": "🚪",
            "STRATEGY": "📈",
            "UTILITY": "🔧",
            "OTHER": "📦"
        }
        
        # Initialize session state
        if "marketplace_initialized" not in st.session_state:
            st.session_state.marketplace_initialized = True
            st.session_state.marketplace_tab = "browse"
            st.session_state.selected_component = None
            st.session_state.search_query = ""
            st.session_state.filtered_type = None
            st.session_state.min_rating = 0.0
            st.session_state.sort_by = "downloads"
            st.session_state.user_components = []
            st.session_state.publish_success = False
            st.session_state.publish_error = None
    
    def render(self):
        """Render the marketplace UI."""
        st.title("Component Marketplace")
        st.write("Share, discover, and use trading components from the community securely.")
        
        # Marketplace tabs
        tabs = ["Browse", "My Components", "Publish", "Settings"]
        selected_tab = st.radio("Marketplace Navigation", tabs, horizontal=True, key="marketplace_nav")
        
        st.session_state.marketplace_tab = selected_tab.lower().replace(" ", "_")
        
        # Render selected tab
        if st.session_state.marketplace_tab == "browse":
            self._render_browse_tab()
        elif st.session_state.marketplace_tab == "my_components":
            self._render_my_components_tab()
        elif st.session_state.marketplace_tab == "publish":
            self._render_publish_tab()
        elif st.session_state.marketplace_tab == "settings":
            self._render_settings_tab()
    
    def _render_browse_tab(self):
        """Render the browse tab."""
        st.header("Browse Components")
        
        # Search and filter controls
        col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
        
        with col1:
            search_query = st.text_input("Search", value=st.session_state.search_query, placeholder="Search components...")
            st.session_state.search_query = search_query
        
        with col2:
            component_types = ["All"] + list(self.component_types.keys())
            selected_type = st.selectbox("Component Type", component_types)
            st.session_state.filtered_type = None if selected_type == "All" else selected_type
        
        with col3:
            min_rating = st.slider("Min Rating", 0.0, 5.0, st.session_state.min_rating, 0.5)
            st.session_state.min_rating = min_rating
        
        with col4:
            sort_options = {
                "downloads": "Downloads",
                "rating": "Rating",
                "published_date": "Newest"
            }
            sort_by = st.selectbox("Sort By", list(sort_options.values()), 
                                 format_func=lambda x: sort_options.get(x, x))
            st.session_state.sort_by = next((k for k, v in sort_options.items() if v == sort_by), "downloads")
        
        # Search for components
        try:
            components = self.api.search_components(
                query=st.session_state.search_query,
                component_type=st.session_state.filtered_type,
                min_rating=st.session_state.min_rating,
                sort_by=st.session_state.sort_by
            )
            
            if not components:
                st.info("No components found matching your criteria.")
                return
            
            # Display components
            self._display_component_grid(components)
            
        except Exception as e:
            st.error(f"Error searching components: {str(e)}")
    
    def _display_component_grid(self, components: List[Dict[str, Any]]):
        """
        Display components in a grid layout.
        
        Args:
            components: List of component metadata
        """
        # Display in a 3-column grid
        cols = st.columns(3)
        
        for i, component in enumerate(components):
            col = cols[i % 3]
            
            with col:
                self._render_component_card(component)
    
    def _render_component_card(self, component: Dict[str, Any]):
        """
        Render a component card.
        
        Args:
            component: Component metadata
        """
        component_id = component.get("component_id", "unknown")
        name = component.get("name", component_id)
        description = component.get("description", "No description provided.")
        version = component.get("version", "1.0.0")
        author = component.get("author", "Anonymous")
        downloads = component.get("downloads", 0)
        rating = component.get("average_rating", 0.0)
        component_type = component.get("component_type", "OTHER")
        
        # Get component type icon
        icon = self.component_types.get(component_type, "📦")
        
        # Card container
        with st.container():
            st.markdown(f"### {icon} {name}")
            st.caption(f"v{version} by {author}")
            
            # Truncate description if too long
            if len(description) > 100:
                st.markdown(f"{description[:100]}...")
            else:
                st.markdown(description)
            
            # Display rating and downloads
            col1, col2 = st.columns(2)
            
            with col1:
                stars = "★" * int(rating) + "☆" * (5 - int(rating))
                st.markdown(f"**Rating:** {stars} ({rating:.1f})")
            
            with col2:
                st.markdown(f"**Downloads:** {downloads}")
            
            # View button
            if st.button("View Details", key=f"view_{component_id}"):
                st.session_state.selected_component = component
                self._show_component_details(component)
    
    def _show_component_details(self, component: Dict[str, Any]):
        """
        Show component details modal.
        
        Args:
            component: Component metadata
        """
        component_id = component.get("component_id", "unknown")
        
        # Fetch full component details if needed
        try:
            details = self.api.get_component_details(component_id, version=component.get("version"))
            
            # Update component data with full details
            component.update(details)
        except Exception as e:
            st.error(f"Error loading component details: {str(e)}")
        
        # Display detailed component information
        st.subheader(f"{component.get('name', component_id)} Details")
        
        tabs = st.tabs(["Overview", "Code", "Reviews", "Versions", "Dependencies"])
        
        with tabs[0]:  # Overview tab
            self._render_component_overview(component)
        
        with tabs[1]:  # Code tab
            self._render_component_code(component)
        
        with tabs[2]:  # Reviews tab
            self._render_component_reviews(component)
        
        with tabs[3]:  # Versions tab
            self._render_component_versions(component)
        
        with tabs[4]:  # Dependencies tab
            self._render_component_dependencies(component)
        
        # Download button
        if st.button("Download and Install", key=f"download_{component_id}"):
            self._download_component(component)
    
    def _render_component_overview(self, component: Dict[str, Any]):
        """
        Render component overview tab.
        
        Args:
            component: Component metadata
        """
        # Component metadata
        st.markdown(f"**Description:** {component.get('description', 'No description provided.')}")
        st.markdown(f"**Author:** {component.get('author', 'Anonymous')}")
        st.markdown(f"**Version:** {component.get('version', '1.0.0')}")
        st.markdown(f"**Type:** {component.get('component_type', 'OTHER')}")
        st.markdown(f"**Published:** {component.get('published_date', 'Unknown')}")
        st.markdown(f"**Downloads:** {component.get('downloads', 0)}")
        
        # Display tags
        tags = component.get("tags", [])
        if tags:
            st.markdown("**Tags:**")
            st.write(" ".join([f"`{tag}`" for tag in tags]))
        
        # Display rating
        rating = component.get("average_rating", 0.0)
        ratings_count = component.get("ratings_count", 0)
        stars = "★" * int(rating) + "☆" * (5 - int(rating))
        st.markdown(f"**Rating:** {stars} ({rating:.1f} from {ratings_count} ratings)")
        
        # Security information
        st.subheader("Security")
        
        verified = component.get("verified", False)
        st.markdown(f"**Verified:** {'✅ Yes' if verified else '❌ No'}")
        
        trust_level = component.get("trust_level", "Untrusted")
        if trust_level == "Trusted":
            st.markdown("**Trust Level:** ✅ Trusted Publisher")
        elif trust_level == "Verified":
            st.markdown("**Trust Level:** ✓ Verified Publisher")
        else:
            st.markdown("**Trust Level:** ⚠️ Untrusted")
        
        signature = component.get("signature")
        if signature:
            st.markdown("**Code Signature:** ✅ Signed")
        else:
            st.markdown("**Code Signature:** ❌ Unsigned")
    
    def _render_component_code(self, component: Dict[str, Any]):
        """
        Render component code tab.
        
        Args:
            component: Component metadata
        """
        component_id = component.get("component_id", "unknown")
        
        try:
            # Fetch code if not already in component data
            if "code" not in component:
                code = self.api.get_component_code(component_id, version=component.get("version"))
                component["code"] = code
            else:
                code = component["code"]
            
            # Display code with syntax highlighting
            st.code(code, language="python")
            
        except Exception as e:
            st.error(f"Error loading component code: {str(e)}")
    
    def _render_component_reviews(self, component: Dict[str, Any]):
        """
        Render component reviews tab.
        
        Args:
            component: Component metadata
        """
        component_id = component.get("component_id", "unknown")
        
        # Get reviews if not already fetched
        try:
            if "reviews" not in component:
                reviews = self.api.get_reviews(component_id, version=component.get("version"))
                component["reviews"] = reviews
            else:
                reviews = component["reviews"]
            
            # Add review form
            st.subheader("Add Review")
            
            user_rating = st.slider("Rating", 1, 5, 5, 1)
            review_text = st.text_area("Review", placeholder="Write your review here...")
            
            if st.button("Submit Review", key=f"submit_review_{component_id}"):
                if not review_text:
                    st.warning("Please write a review before submitting.")
                else:
                    # Generate a unique review ID
                    review_id = str(uuid.uuid4())
                    
                    # Submit rating and review
                    try:
                        # Add rating
                        self.api.rate_component(
                            component_id=component_id,
                            rating=user_rating,
                            user_id="current_user",  # Replace with actual user ID
                            version=component.get("version")
                        )
                        
                        # Add review
                        self.api.add_review(
                            component_id=component_id,
                            review_id=review_id,
                            review_text=review_text,
                            user_id="current_user",  # Replace with actual user ID
                            version=component.get("version")
                        )
                        
                        st.success("Review submitted successfully.")
                        
                        # Refresh reviews
                        component["reviews"] = self.api.get_reviews(
                            component_id=component_id,
                            version=component.get("version")
                        )
                        reviews = component["reviews"]
                        
                    except Exception as e:
                        st.error(f"Error submitting review: {str(e)}")
            
            # Display existing reviews
            st.subheader("Reviews")
            
            if not reviews:
                st.info("No reviews yet. Be the first to review this component!")
            else:
                for review in reviews:
                    with st.container():
                        # Review header
                        user_id = review.get("user_id", "Anonymous")
                        timestamp = review.get("timestamp", "Unknown date")
                        
                        # Parse and format timestamp
                        try:
                            dt = datetime.fromisoformat(timestamp)
                            formatted_date = dt.strftime("%Y-%m-%d %H:%M")
                        except:
                            formatted_date = timestamp
                        
                        st.markdown(f"**{user_id}** - {formatted_date}")
                        
                        # Display stars for user's rating
                        user_rating = review.get("rating", 0)
                        stars = "★" * int(user_rating) + "☆" * (5 - int(user_rating))
                        st.markdown(f"{stars}")
                        
                        # Review text
                        st.markdown(review.get("text", "No text provided."))
                        
                        # Helpful button
                        helpful_count = review.get("helpful_count", 0)
                        st.button(f"Helpful ({helpful_count})", key=f"helpful_{review.get('review_id', uuid.uuid4())}")
                        
                        st.markdown("---")
        
        except Exception as e:
            st.error(f"Error loading reviews: {str(e)}")
    
    def _render_component_versions(self, component: Dict[str, Any]):
        """
        Render component versions tab.
        
        Args:
            component: Component metadata
        """
        component_id = component.get("component_id", "unknown")
        current_version = component.get("version", "1.0.0")
        
        try:
            # Get all versions if not already fetched
            if "versions" not in component:
                versions = self.api.get_component_versions(component_id)
                component["versions"] = versions
            else:
                versions = component["versions"]
            
            if not versions:
                st.info(f"No version history available for this component.")
                return
            
            # Display version history
            st.subheader("Version History")
            
            # Create a table of versions
            version_data = []
            for version in versions:
                version_num = version.get("version", "Unknown")
                published_date = version.get("published_date", "Unknown")
                changes = version.get("changes", "No change notes provided.")
                
                version_data.append({
                    "Version": version_num,
                    "Published": published_date,
                    "Changes": changes,
                    "Current": version_num == current_version
                })
            
            # Convert to DataFrame for display
            df = pd.DataFrame(version_data)
            st.dataframe(df)
            
        except Exception as e:
            st.error(f"Error loading version history: {str(e)}")
    
    def _render_component_dependencies(self, component: Dict[str, Any]):
        """
        Render component dependencies tab.
        
        Args:
            component: Component metadata
        """
        # Get dependencies
        dependencies = component.get("dependencies", [])
        
        if not dependencies:
            st.info("This component has no dependencies.")
            return
        
        # Display dependencies
        st.subheader("Dependencies")
        
        for dep in dependencies:
            dep_id = dep.get("component_id", "Unknown")
            version_req = dep.get("version_requirement", "Any")
            
            with st.container():
                st.markdown(f"**{dep_id}** - Required version: {version_req}")
                
                # Check if dependency is installed
                try:
                    installed = self.api.is_component_installed(dep_id)
                    if installed:
                        installed_version = self.api.get_installed_version(dep_id)
                        compatible = self.api.check_version_compatibility(
                            installed_version, version_req
                        )
                        
                        if compatible:
                            st.markdown("✅ Installed (Compatible)")
                        else:
                            st.markdown(f"⚠️ Installed version {installed_version} is not compatible")
                            if st.button(f"Install Compatible Version", key=f"install_dep_{dep_id}"):
                                self._install_dependency(dep_id, version_req)
                    else:
                        st.markdown("❌ Not installed")
                        if st.button(f"Install Dependency", key=f"install_dep_{dep_id}"):
                            self._install_dependency(dep_id, version_req)
                
                except Exception as e:
                    st.error(f"Error checking dependency: {str(e)}")
    
    def _download_component(self, component: Dict[str, Any]):
        """
        Download and install a component.
        
        Args:
            component: Component metadata
        """
        component_id = component.get("component_id", "unknown")
        version = component.get("version", "1.0.0")
        
        try:
            # Download component
            st.info(f"Downloading component {component_id} v{version}...")
            
            # Check dependencies first
            dependencies = component.get("dependencies", [])
            missing_deps = []
            
            for dep in dependencies:
                dep_id = dep.get("component_id")
                version_req = dep.get("version_requirement")
                
                if not self.api.is_component_installed(dep_id):
                    missing_deps.append((dep_id, version_req))
                else:
                    installed_version = self.api.get_installed_version(dep_id)
                    if not self.api.check_version_compatibility(installed_version, version_req):
                        missing_deps.append((dep_id, version_req))
            
            # If missing dependencies, ask to install them
            if missing_deps:
                st.warning(f"This component has {len(missing_deps)} missing or incompatible dependencies.")
                
                for dep_id, version_req in missing_deps:
                    st.markdown(f"- {dep_id} (required: {version_req})")
                
                if st.button("Install Dependencies and Component"):
                    # Install dependencies first
                    for dep_id, version_req in missing_deps:
                        self._install_dependency(dep_id, version_req)
                    
                    # Then install the component
                    result = self.api.download_component(component_id, version=version)
                    
                    if result.get("success"):
                        st.success(f"Component {component_id} v{version} installed successfully!")
                    else:
                        st.error(f"Failed to install component: {result.get('error')}")
            else:
                # No missing dependencies, install directly
                result = self.api.download_component(component_id, version=version)
                
                if result.get("success"):
                    st.success(f"Component {component_id} v{version} installed successfully!")
                else:
                    st.error(f"Failed to install component: {result.get('error')}")
        
        except Exception as e:
            st.error(f"Error downloading component: {str(e)}")
    
    def _install_dependency(self, component_id: str, version_requirement: str):
        """
        Install a dependency component.
        
        Args:
            component_id: Component ID
            version_requirement: Version requirement
        """
        try:
            st.info(f"Installing dependency {component_id} ({version_requirement})...")
            
            # Find compatible version
            versions = self.api.find_compatible_versions(component_id, version_requirement)
            
            if not versions:
                st.error(f"No compatible version found for {component_id} ({version_requirement})")
                return
            
            # Get latest compatible version
            latest_compatible = versions[0]
            
            # Download and install
            result = self.api.download_component(component_id, version=latest_compatible)
            
            if result.get("success"):
                st.success(f"Dependency {component_id} v{latest_compatible} installed successfully!")
            else:
                st.error(f"Failed to install dependency: {result.get('error')}")
        
        except Exception as e:
            st.error(f"Error installing dependency: {str(e)}")

    def _render_my_components_tab(self):
        """Render the my components tab."""
        st.header("My Components")
        
        # Load user's components
        try:
            user_components = self.api.get_user_components()
            
            if not user_components:
                st.info("You haven't published any components yet.")
                st.write("You can publish your trading components by switching to the 'Publish' tab.")
                return
            
            # Store in session state
            st.session_state.user_components = user_components
            
            # Display user components
            for component in user_components:
                with st.container():
                    self._render_user_component_card(component)
                    st.markdown("---")
        
        except Exception as e:
            st.error(f"Error loading your components: {str(e)}")
    
    def _render_user_component_card(self, component: Dict[str, Any]):
        """
        Render a user component card.
        
        Args:
            component: Component metadata
        """
        component_id = component.get("component_id", "unknown")
        name = component.get("name", component_id)
        version = component.get("version", "1.0.0")
        downloads = component.get("downloads", 0)
        rating = component.get("average_rating", 0.0)
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"### {name}")
            st.caption(f"v{version}")
            st.markdown(component.get("description", "No description provided."))
        
        with col2:
            st.markdown(f"**Downloads:** {downloads}")
            stars = "★" * int(rating) + "☆" * (5 - int(rating))
            st.markdown(f"**Rating:** {stars} ({rating:.1f})")
        
        with col3:
            st.button("View Analytics", key=f"analytics_{component_id}")
            st.button("Update Component", key=f"update_{component_id}")
            if st.button("Delete", key=f"delete_{component_id}"):
                self._delete_component(component_id)
    
    def _delete_component(self, component_id: str):
        """
        Delete a component.
        
        Args:
            component_id: Component ID
        """
        if st.session_state.get("confirm_delete") != component_id:
            st.session_state.confirm_delete = component_id
            st.warning(f"Are you sure you want to delete this component? This action cannot be undone.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, Delete", key=f"confirm_delete_{component_id}"):
                    try:
                        self.api.delete_component(component_id)
                        st.success(f"Component {component_id} deleted successfully.")
                        st.session_state.confirm_delete = None
                        
                        # Refresh user components
                        st.session_state.user_components = self.api.get_user_components()
                    except Exception as e:
                        st.error(f"Error deleting component: {str(e)}")
            
            with col2:
                if st.button("Cancel", key=f"cancel_delete_{component_id}"):
                    st.session_state.confirm_delete = None
    
    def _render_publish_tab(self):
        """Render the publish tab."""
        st.header("Publish Component")
        st.write("Share your trading components with the community.")
        
        # Check if previous publish was successful
        if st.session_state.publish_success:
            st.success("Component published successfully!")
            if st.button("Clear Success Message"):
                st.session_state.publish_success = False
        
        if st.session_state.publish_error:
            st.error(f"Error publishing component: {st.session_state.publish_error}")
            if st.button("Clear Error Message"):
                st.session_state.publish_error = None
        
        # Component upload form
        with st.form("publish_form"):
            # Basic information
            component_name = st.text_input("Component Name*", placeholder="e.g., MovingAverageCross")
            component_type = st.selectbox("Component Type*", list(self.component_types.keys()))
            description = st.text_area("Description*", placeholder="Describe your component...")
            
            # Version info
            version = st.text_input("Version*", value="1.0.0", placeholder="1.0.0")
            changes = st.text_area("Changes", placeholder="What's new in this version?")
            
            # Tags
            tags_input = st.text_input("Tags (comma-separated)", placeholder="trend, momentum, volatility")
            
            # Dependencies
            dependencies_input = st.text_area(
                "Dependencies (one per line, format: component_id>=version)", 
                placeholder="ExponentialMovingAverage>=1.0.0"
            )
            
            # Code upload
            upload_col, code_col = st.columns(2)
            
            with upload_col:
                uploaded_file = st.file_uploader("Upload Component File", type=["py"])
                
                if uploaded_file:
                    st.success(f"File {uploaded_file.name} uploaded.")
            
            with code_col:
                code_content = st.text_area("Or paste code here", height=300)
            
            # Terms and conditions
            terms_accepted = st.checkbox("I accept the terms and conditions for publishing components")
            license_type = st.selectbox("License", [
                "MIT License", 
                "Apache License 2.0", 
                "GNU General Public License v3.0",
                "BSD 3-Clause License",
                "Custom License"
            ])
            
            # Submit button
            submit_button = st.form_submit_button("Publish Component")
            
            if submit_button:
                # Validate required fields
                if not (component_name and component_type and description and version):
                    st.error("Please fill in all required fields (marked with *).")
                    return
                
                if not terms_accepted:
                    st.error("You must accept the terms and conditions to publish components.")
                    return
                
                if not (uploaded_file or code_content):
                    st.error("Please either upload a file or paste your code.")
                    return
                
                # Process code content
                if uploaded_file:
                    code_bytes = uploaded_file.getvalue()
                    code_str = code_bytes.decode("utf-8")
                else:
                    code_str = code_content
                    code_bytes = code_str.encode("utf-8")
                
                # Process tags
                tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]
                
                # Process dependencies
                dependencies = []
                if dependencies_input:
                    for line in dependencies_input.strip().split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                            
                        # Parse dependency line
                        try:
                            # Format should be component_id>=version or similar
                            import re
                            match = re.match(r"([a-zA-Z0-9_]+)([>=<]+)(.+)", line)
                            
                            if match:
                                dep_id, operator, dep_version = match.groups()
                                dependencies.append({
                                    "component_id": dep_id,
                                    "version_requirement": f"{operator}{dep_version}"
                                })
                            else:
                                # If no operator, assume exact version
                                if ":" in line:
                                    dep_id, dep_version = line.split(":", 1)
                                else:
                                    dep_id, dep_version = line, "1.0.0"
                                    
                                dependencies.append({
                                    "component_id": dep_id.strip(),
                                    "version_requirement": f"=={dep_version.strip()}"
                                })
                        except Exception as e:
                            st.error(f"Invalid dependency format: {line} - {str(e)}")
                            return
                
                # Generate component ID from name if not provided
                component_id = component_name.replace(" ", "_").lower()
                
                # Create metadata
                metadata = {
                    "component_id": component_id,
                    "name": component_name,
                    "component_type": component_type,
                    "description": description,
                    "version": version,
                    "author": "current_user",  # Replace with actual user ID
                    "published_date": datetime.now().isoformat(),
                    "changes": changes,
                    "tags": tags,
                    "dependencies": dependencies,
                    "license": license_type
                }
                
                # Publish component
                try:
                    result = self.api.publish_component(
                        component_id=component_id,
                        component_data=code_bytes,
                        metadata=metadata
                    )
                    
                    if result.get("success"):
                        st.session_state.publish_success = True
                        st.session_state.publish_error = None
                    else:
                        st.session_state.publish_error = result.get("error", "Unknown error")
                        st.session_state.publish_success = False
                
                except Exception as e:
                    st.session_state.publish_error = str(e)
                    st.session_state.publish_success = False
    
    def _render_settings_tab(self):
        """Render the settings tab."""
        st.header("Marketplace Settings")
        
        # Security settings
        st.subheader("Security Settings")
        
        # API key management
        st.markdown("### API Key Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate New API Key"):
                try:
                    api_key = self.api.security_manager.generate_api_key("current_user")
                    st.code(api_key)
                    st.warning("Save this API key securely. It won't be shown again.")
                except Exception as e:
                    st.error(f"Error generating API key: {str(e)}")
        
        with col2:
            if st.button("Revoke All API Keys"):
                try:
                    self.api.security_manager.revoke_api_keys("current_user")
                    st.success("All API keys revoked successfully.")
                except Exception as e:
                    st.error(f"Error revoking API keys: {str(e)}")
        
        # Trusted publishers
        st.markdown("### Trusted Publishers")
        
        trusted_publishers = self.api.security_manager.get_trusted_publishers()
        
        if not trusted_publishers:
            st.info("You don't have any trusted publishers yet.")
        else:
            for publisher in trusted_publishers:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**{publisher}**")
                
                with col2:
                    if st.button("Remove", key=f"remove_{publisher}"):
                        try:
                            self.api.security_manager.remove_trusted_publisher(publisher)
                            st.success(f"Publisher {publisher} removed from trusted list.")
                        except Exception as e:
                            st.error(f"Error removing trusted publisher: {str(e)}")
        
        # Add trusted publisher
        new_publisher = st.text_input("Add Trusted Publisher", placeholder="Publisher ID")
        if st.button("Add Publisher") and new_publisher:
            try:
                self.api.security_manager.add_trusted_publisher(new_publisher)
                st.success(f"Publisher {new_publisher} added to trusted list.")
            except Exception as e:
                st.error(f"Error adding trusted publisher: {str(e)}")
        
        # Verification settings
        st.markdown("### Verification Settings")
        
        verification_options = [
            "Always verify all downloaded components",
            "Verify only untrusted components",
            "Skip verification (not recommended)"
        ]
        verification_setting = st.radio("Verification Policy", verification_options)
        
        # Execution settings
        st.markdown("### Execution Settings")
        
        execution_options = [
            "Always use sandbox for all components",
            "Use sandbox only for untrusted components",
            "Never use sandbox (not recommended)"
        ]
        execution_setting = st.radio("Component Execution Policy", execution_options)
        
        # Save settings
        if st.button("Save Settings"):
            # Convert settings to API configuration
            verify_policy = "all"
            if verification_setting == verification_options[1]:
                verify_policy = "untrusted"
            elif verification_setting == verification_options[2]:
                verify_policy = "none"
            
            sandbox_policy = "all"
            if execution_setting == execution_options[1]:
                sandbox_policy = "untrusted"
            elif execution_setting == execution_options[2]:
                sandbox_policy = "none"
            
            # Save settings
            try:
                self.api.update_settings({
                    "verify_policy": verify_policy,
                    "sandbox_policy": sandbox_policy
                })
                st.success("Settings saved successfully.")
            except Exception as e:
                st.error(f"Error saving settings: {str(e)}")
