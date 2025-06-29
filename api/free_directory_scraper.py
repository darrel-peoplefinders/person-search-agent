"""
Free Directory Sites Scraper
Searches PeopleFinders network sites for free limited profiles
"""

import asyncio
import httpx
from typing import List, Dict, Optional
from urllib.parse import quote
import re

class FreeDirectoryResult:
    def __init__(self, site_name: str, site_url: str, person_name: str, 
                 location: str, snippet: str, profile_url: str):
        self.site_name = site_name
        self.site_url = site_url  
        self.person_name = person_name
        self.location = location
        self.snippet = snippet
        self.profile_url = profile_url

class FreeDirectoryScraper:
    """Scraper for PeopleFinders network free directory sites"""
    
    NETWORK_SITES = [
        {
            "name": "FastPeopleSearch",
            "base_url": "https://www.fastpeoplesearch.com",
            "search_path": "/name/{first_name}-{last_name}_{state}",
            "description": "Standard directory data"
        },
        {
            "name": "TruePeopleSearch", 
            "base_url": "https://www.truepeoplesearch.com",
            "search_path": "/results?name={first_name}%20{last_name}&citystatezip={state}",
            "description": "Comprehensive profiles"
        },
        {
            "name": "USPhoneBook",
            "base_url": "https://www.usphonebook.com",
            "search_path": "/{first_name}-{last_name}/{state}",
            "description": "Basic contact information"  
        },
        {
            "name": "SearchPeopleFree",
            "base_url": "https://www.searchpeoplefree.com",
            "search_path": "/find/{first_name}-{last_name}/{state}",
            "description": "General directory"
        }
    ]
    
    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )
    
    async def search_all_sites(self, first_name: str, last_name: str, 
                              state: str = None) -> List[FreeDirectoryResult]:
        """
        Search all network sites for a person
        
        Args:
            first_name: Person's first name
            last_name: Person's last name  
            state: State abbreviation (optional)
            
        Returns:
            List of FreeDirectoryResult objects
        """
        results = []
        
        # Search each site concurrently
        tasks = []
        for site in self.NETWORK_SITES:
            task = self._search_site(site, first_name, last_name, state)
            tasks.append(task)
        
        site_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for site_result in site_results:
            if isinstance(site_result, Exception):
                continue  # Skip failed searches
            if site_result:
                results.extend(site_result)
        
        return results
    
    async def _search_site(self, site: Dict, first_name: str, last_name: str, 
                          state: str = None) -> List[FreeDirectoryResult]:
        """Search a single directory site"""
        try:
            # Build search URL
            search_path = site["search_path"].format(
                first_name=quote(first_name.lower()),
                last_name=quote(last_name.lower()),
                state=state.upper() if state else "US"
            )
            search_url = site["base_url"] + search_path
            
            # Make request
            response = await self.client.get(search_url)
            if response.status_code != 200:
                return []
            
            # Parse results (simplified - in production would need site-specific parsing)
            results = self._parse_site_results(
                site, response.text, first_name, last_name, state
            )
            
            return results
            
        except Exception as e:
            print(f"Error searching {site['name']}: {e}")
            return []
    
    def _parse_site_results(self, site: Dict, html: str, first_name: str, 
                           last_name: str, state: str = None) -> List[FreeDirectoryResult]:
        """Parse HTML results from a directory site"""
        results = []
        
        # This is a simplified parser - in production would need proper HTML parsing
        # for each site's specific structure
        
        # Look for the person's name in the HTML (case insensitive)
        name_pattern = rf"{re.escape(first_name)}.*?{re.escape(last_name)}"
        name_matches = re.findall(name_pattern, html, re.IGNORECASE)
        
        if name_matches:
            # Create a mock result for demonstration
            # In production, would extract actual profile URL and snippet
            result = FreeDirectoryResult(
                site_name=site["name"],
                site_url=site["base_url"],
                person_name=f"{first_name} {last_name}",
                location=f"{state or 'Multiple States'}",
                snippet=f"***{first_name} {last_name}*** in {state or 'US'}. {site['description']}. View full profile.",
                profile_url=site["base_url"] + f"/profile/{first_name.lower()}-{last_name.lower()}"
            )
            results.append(result)
        
        return results
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

# Helper function to generate free directory results for search response
def generate_free_directory_links(first_name: str, last_name: str, 
                                 state: str = None) -> List[Dict[str, str]]:
    """
    Generate free directory links for display in search results
    This is a fast synchronous version for immediate display
    """
    links = []
    
    sites = [
        {
            "name": "FastPeopleSearch",
            "description": "Standard", 
            "url": f"https://www.fastpeoplesearch.com/name/{first_name.lower()}-{last_name.lower()}_{state or 'US'}"
        },
        {
            "name": "TruePeopleSearch",
            "description": "Comprehensive",
            "url": f"https://www.truepeoplesearch.com/results?name={first_name}%20{last_name}&citystatezip={state or ''}"
        },
        {
            "name": "USPhoneBook", 
            "description": "Basic Contact",
            "url": f"https://www.usphonebook.com/{first_name.lower()}-{last_name.lower()}/{state or 'US'}"
        }
    ]
    
    for site in sites:
        snippet = f"***{first_name} {last_name}*** in {state or 'Multiple States'}. {site['description']} directory information available."
        
        links.append({
            "site_name": site["name"],
            "description": site["description"], 
            "snippet": snippet,
            "url": site["url"]
        })
    
    return links