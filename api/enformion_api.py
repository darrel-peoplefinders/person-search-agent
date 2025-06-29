"""
EnformionGo API Integration
This module handles the connection to EnformionGo person search API
"""

import httpx
import asyncio
from typing import Optional, Dict, List, Any
from pydantic import BaseModel
import os
from datetime import datetime

class EnformionConfig(BaseModel):
    """Configuration for EnformionGo API"""
    api_key: str
    ap_name: str  # galaxy-ap-name
    ap_password: str  # galaxy-ap-password
    base_url: str = "https://devapi.enformion.com"
    timeout: int = 30
    max_results: int = 10
    search_type: str = "Teaser"  # Person, Teaser, ReversePhonePersonTeaser, ReversePhonePerson
    client_type: str = "API"

class PersonSearchParams(BaseModel):
    """Parameters for person search"""
    first_name: str
    last_name: str
    middle_name: Optional[str] = None
    age: Optional[int] = None
    age_range_min: Optional[int] = None
    age_range_max: Optional[int] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    previous_locations: Optional[List[str]] = None

class EnformionResult(BaseModel):
    """Single result from EnformionGo API"""
    id: str
    confidence_score: float
    first_name: str
    last_name: str
    middle_name: Optional[str] = None
    age: Optional[int] = None
    current_address: Optional[Dict[str, str]] = None
    previous_addresses: Optional[List[Dict[str, str]]] = None
    phone_numbers: Optional[List[str]] = None
    email_addresses: Optional[List[str]] = None
    relatives: Optional[List[Dict[str, str]]] = None
    associates: Optional[List[Dict[str, str]]] = None
    employment_history: Optional[List[Dict[str, str]]] = None
    education_history: Optional[List[Dict[str, str]]] = None
    criminal_records: Optional[List[Dict[str, Any]]] = None
    bankruptcy_records: Optional[List[Dict[str, Any]]] = None
    last_updated: Optional[datetime] = None

class EnformionResponse(BaseModel):
    """Response from EnformionGo API"""
    success: bool
    total_results: int
    results: List[EnformionResult]
    query_id: str
    credits_used: int
    error_message: Optional[str] = None

class EnformionClient:
    """Client for EnformionGo API"""
    
    def __init__(self, config: EnformionConfig):
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=config.timeout,
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "galaxy-ap-name": config.ap_name,
                "galaxy-ap-password": config.ap_password,
                "galaxy-search-type": config.search_type,
                "galaxy-client-type": config.client_type,
                "galaxy-client-session-id": f"session_{int(datetime.now().timestamp())}"
            }
        )
    
    async def search_person(self, params: PersonSearchParams) -> EnformionResponse:
        """
        Search for a person using EnformionGo API
        
        Args:
            params: Search parameters
            
        Returns:
            EnformionResponse with search results
        """
        try:
            # Build EnformionGo API request payload
            payload = {
                "FirstName": params.first_name,
                "LastName": params.last_name,
                "Page": 1,
                "ResultsPerPage": self.config.max_results,
                "Includes": [
                    "Addresses", 
                    "PhoneNumbers", 
                    "EmailAddresses", 
                    "DatesOfBirth",
                    "RelativesSummary",
                    "AssociatesSummary"
                ]
            }
            
            # Add optional parameters
            if params.middle_name:
                payload["MiddleName"] = params.middle_name
            if params.age:
                payload["Age"] = params.age
            elif params.age_range_min and params.age_range_max:
                payload["AgeRangeMinAge"] = params.age_range_min
                payload["AgeRangeMaxAge"] = params.age_range_max
                
            # Add location information
            if params.city or params.state or params.zip_code:
                addresses = []
                address_line2_parts = []
                
                if params.city:
                    address_line2_parts.append(params.city)
                if params.state:
                    address_line2_parts.append(params.state)
                if params.zip_code:
                    address_line2_parts.append(params.zip_code)
                    
                if address_line2_parts:
                    addresses.append({
                        "AddressLine2": ", ".join(address_line2_parts)
                    })
                    payload["Addresses"] = addresses
            
            # Make API request to EnformionGo
            response = await self.client.post(
                f"{self.config.base_url}/PersonSearch",
                json=payload
            )
            
            print(f"EnformionGo Response Status: {response.status_code}")
            print(f"EnformionGo Response Headers: {response.headers}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"EnformionGo Response Data Type: {type(data)}")
                print(f"EnformionGo Response Length: {len(data) if isinstance(data, list) else 'Not a list'}")
                if isinstance(data, list) and len(data) > 0:
                    print(f"First Record Keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not a dict'}")
                elif isinstance(data, dict):
                    print(f"Response Dict Keys: {list(data.keys())}")
                return self._parse_enformion_response(data)
            else:
                return EnformionResponse(
                    success=False,
                    total_results=0,
                    results=[],
                    query_id="",
                    credits_used=0,
                    error_message=f"EnformionGo API request failed: {response.status_code} - {response.text}"
                )
                
        except httpx.TimeoutException:
            return EnformionResponse(
                success=False,
                total_results=0,
                results=[],
                query_id="",
                credits_used=0,
                error_message="Request timeout - EnformionGo API did not respond in time"
            )
        except Exception as e:
            return EnformionResponse(
                success=False,
                total_results=0,
                results=[],
                query_id="",
                credits_used=0,
                error_message=f"Unexpected error calling EnformionGo API: {str(e)}"
            )
    
    def _parse_enformion_response(self, data: Dict[str, Any]) -> EnformionResponse:
        """Parse EnformionGo API response into EnformionResponse object"""
        try:
            results = []
            
            # EnformionGo returns a dict with 'persons' key containing the records
            print(f"Parsing response with keys: {list(data.keys())}")
            
            # Extract the persons array from the response
            records = data.get("persons", [])
            print(f"Found {len(records)} person records")
            
            for i, record in enumerate(records):
                if not isinstance(record, dict):
                    continue
                    
                print(f"Processing record {i+1} with keys: {list(record.keys())}")
                
                # Extract basic info - the actual field structure from your sample
                first_name = ""
                last_name = ""
                middle_name = ""
                
                # Try different possible name field structures
                if "name" in record:
                    name_obj = record["name"]
                    first_name = name_obj.get("firstName", "")
                    last_name = name_obj.get("lastName", "")
                    middle_name = name_obj.get("middleName", "")
                    print(f"Found name: {first_name} {last_name}")
                elif "fullName" in record:
                    # Sometimes it's in fullName field
                    full_name = record.get("fullName", "")
                    name_parts = full_name.split()
                    if len(name_parts) >= 2:
                        first_name = name_parts[0]
                        last_name = name_parts[-1]
                        if len(name_parts) > 2:
                            middle_name = " ".join(name_parts[1:-1])
                    print(f"Parsed fullName: {first_name} {last_name}")
                
                # Extract age
                age = record.get("age", 0)
                print(f"Found age: {age}")
                
                # Extract addresses from the actual structure
                addresses = record.get("addresses", [])
                current_address = None
                previous_addresses = []
                
                if addresses:
                    print(f"Found {len(addresses)} addresses")
                    # Sort by addressOrder to get most recent first
                    sorted_addresses = sorted(addresses, key=lambda x: x.get("addressOrder", 999))
                    
                    for j, addr in enumerate(sorted_addresses[:5]):  # Limit to 5 addresses
                        address_data = {
                            "city": addr.get("city", ""),
                            "state": addr.get("state", ""),
                            "zip": addr.get("zip", ""),
                            "address": addr.get("fullAddress", "")
                        }
                        
                        if j == 0:
                            current_address = address_data
                        else:
                            previous_addresses.append(address_data)
                
                # Extract phone numbers
                phones = record.get("phoneNumbers", [])
                phone_list = []
                for phone in phones[:5]:  # Limit to 5 phones
                    phone_num = phone.get("phoneNumber", "")
                    if phone_num:
                        phone_list.append(phone_num)
                
                print(f"Found {len(phone_list)} phone numbers")
                
                # Extract emails
                emails = record.get("emailAddresses", [])
                email_list = []
                for email in emails[:3]:  # Limit to 3 emails
                    email_addr = email.get("emailAddress", "")
                    if email_addr:
                        email_list.append(email_addr)
                
                print(f"Found {len(email_list)} email addresses")
                
                # Extract relatives - try different possible field names
                relatives_list = []
                relatives_data = record.get("relativesSummary", []) or record.get("relatives", [])
                for rel in relatives_data[:5]:  # Limit to 5 relatives
                    if isinstance(rel, dict):
                        # Try different name structures
                        rel_first = ""
                        rel_last = ""
                        
                        if "name" in rel:
                            rel_name = rel["name"]
                            rel_first = rel_name.get("firstName", "")
                            rel_last = rel_name.get("lastName", "")
                        elif "fullName" in rel:
                            full_name = rel.get("fullName", "")
                            name_parts = full_name.split()
                            if len(name_parts) >= 2:
                                rel_first = name_parts[0]
                                rel_last = name_parts[-1]
                        
                        if rel_first or rel_last:
                            relatives_list.append({"first_name": rel_first, "last_name": rel_last})
                
                # Calculate confidence score based on data richness
                confidence = 0.6  # Base confidence
                if current_address and current_address["city"]:
                    confidence += 0.1
                if phone_list:
                    confidence += 0.1
                if email_list:
                    confidence += 0.1
                if age > 0:
                    confidence += 0.05
                if relatives_list:
                    confidence += 0.05
                
                # Only add result if we have at least a name
                if first_name or last_name:
                    result = EnformionResult(
                        id=record.get("tahoeId", f"enf_{len(results)}"),
                        confidence_score=min(confidence, 1.0),
                        first_name=first_name,
                        last_name=last_name,
                        middle_name=middle_name,
                        age=age if age > 0 else None,
                        current_address=current_address,
                        previous_addresses=previous_addresses,
                        phone_numbers=phone_list,
                        email_addresses=email_list,
                        relatives=relatives_list
                    )
                    results.append(result)
                    print(f"Added result: {first_name} {last_name}")
            
            print(f"Final results count: {len(results)}")
            
            return EnformionResponse(
                success=True,
                total_results=len(results),
                results=results,
                query_id=data.get("requestId", f"query_{int(datetime.now().timestamp())}"),
                credits_used=1
            )
            
        except Exception as e:
            print(f"Parse error: {e}")  # Debug logging
            import traceback
            traceback.print_exc()
            return EnformionResponse(
                success=False,
                total_results=0,
                results=[],
                query_id="",
                credits_used=0,
                error_message=f"Failed to parse EnformionGo response: {str(e)}"
            )
        """Parse API response into EnformionResponse object"""
        try:
            results = []
            for result_data in data.get("results", []):
                result = EnformionResult(
                    id=result_data.get("id", ""),
                    confidence_score=result_data.get("confidence_score", 0.0),
                    first_name=result_data.get("first_name", ""),
                    last_name=result_data.get("last_name", ""),
                    middle_name=result_data.get("middle_name"),
                    age=result_data.get("age"),
                    current_address=result_data.get("current_address"),
                    previous_addresses=result_data.get("previous_addresses", []),
                    phone_numbers=result_data.get("phone_numbers", []),
                    email_addresses=result_data.get("email_addresses", []),
                    relatives=result_data.get("relatives", []),
                    associates=result_data.get("associates", []),
                    employment_history=result_data.get("employment_history", []),
                    education_history=result_data.get("education_history", []),
                    criminal_records=result_data.get("criminal_records", []),
                    bankruptcy_records=result_data.get("bankruptcy_records", [])
                )
                results.append(result)
            
            return EnformionResponse(
                success=True,
                total_results=data.get("total_results", len(results)),
                results=results,
                query_id=data.get("query_id", ""),
                credits_used=data.get("credits_used", 0)
            )
            
        except Exception as e:
            return EnformionResponse(
                success=False,
                total_results=0,
                results=[],
                query_id="",
                credits_used=0,
                error_message=f"Failed to parse response: {str(e)}"
            )
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

# =========================
# Factory Functions
# =========================

def create_enformion_client() -> Optional[EnformionClient]:
    """
    Create EnformionGo client from environment variables
    
    Environment variables needed:
    - ENFORMION_AP_NAME: Your EnformionGo Access Profile Name
    - ENFORMION_AP_PASSWORD: Your EnformionGo Access Profile Password
    - ENFORMION_BASE_URL: API base URL (optional)
    - ENFORMION_SEARCH_TYPE: Search type (optional, defaults to 'Teaser')
    """
    ap_name = os.getenv("ENFORMION_AP_NAME")
    ap_password = os.getenv("ENFORMION_AP_PASSWORD")
    
    if not ap_name or not ap_password:
        return None
    
    config = EnformionConfig(
        api_key=os.getenv("ENFORMION_API_KEY", ap_password),  # Use password as fallback
        ap_name=ap_name,
        ap_password=ap_password,
        base_url=os.getenv("ENFORMION_BASE_URL", "https://devapi.enformion.com"),
        timeout=int(os.getenv("ENFORMION_TIMEOUT", "30")),
        max_results=int(os.getenv("ENFORMION_MAX_RESULTS", "10")),
        search_type=os.getenv("ENFORMION_SEARCH_TYPE", "Teaser"),
        client_type=os.getenv("ENFORMION_CLIENT_TYPE", "API")
    )
    
    return EnformionClient(config)

# =========================
# Conversion Helpers
# =========================

def enformion_to_search_result(enformion_result: EnformionResult, search_context: dict = None) -> Dict[str, Any]:
    """
    Convert EnformionResult to the format expected by the person search agent
    """
    # Extract current location
    current_city = ""
    current_state = ""
    if enformion_result.current_address:
        current_city = enformion_result.current_address.get("city", "")
        current_state = enformion_result.current_address.get("state", "")
    
    # Extract previous locations
    previous_locations = []
    if enformion_result.previous_addresses:
        for addr in enformion_result.previous_addresses:
            city = addr.get("city", "")
            state = addr.get("state", "")
            if city and state:
                previous_locations.append(f"{city}, {state}")
    
    # Extract relatives names
    relatives = []
    if enformion_result.relatives:
        for rel in enformion_result.relatives:
            name = f"{rel.get('first_name', '')} {rel.get('last_name', '')}".strip()
            if name:
                relatives.append(name)
    
    # Generate story-based timeline analysis
    timeline_match = _generate_timeline_analysis(enformion_result, search_context)
    
    # Extract professional background
    professional_background = ""
    if enformion_result.employment_history:
        latest_job = enformion_result.employment_history[0]
        company = latest_job.get("company", "")
        title = latest_job.get("title", "")
        if company or title:
            professional_background = f"{title} at {company}".strip(" at ")
    
    return {
        "id": enformion_result.id,
        "confidence": int(enformion_result.confidence_score * 100),
        "first_name": enformion_result.first_name,
        "last_name": enformion_result.last_name,
        "age": enformion_result.age or 0,
        "city": current_city,
        "state": current_state,
        "previous_addresses": previous_locations,
        "relatives": relatives,
        "timeline_match": timeline_match,
        "professional_background": professional_background,
        "phone_numbers": enformion_result.phone_numbers or [],
        "email_addresses": enformion_result.email_addresses or []
    }

def _generate_timeline_analysis(result: EnformionResult, search_context: dict = None) -> str:
    """
    Generate a timeline analysis string that tells the story of this person
    matching the user's search context
    """
    analysis_parts = []
    
    # Extract search context for story-telling
    relationship = search_context.get("relationship", "") if search_context else ""
    additional_context = search_context.get("additional_context", "") if search_context else ""
    first_name = search_context.get("first_name", "") if search_context else ""
    
    # Base age/timeline analysis
    if result.age:
        analysis_parts.append(f"At age {result.age}, this matches your expected timeline")
    
    # Story-based geographic analysis
    if result.current_address and result.previous_addresses:
        current_state = result.current_address.get("state", "")
        current_city = result.current_address.get("city", "")
        
        # Look for meaningful patterns in address history
        prev_states = [addr.get("state") for addr in result.previous_addresses if addr.get("state")]
        prev_cities = [addr.get("city") for addr in result.previous_addresses if addr.get("city")]
        
        # Build the story
        if "college" in relationship.lower() and "2010" in additional_context:
            if "CA" in prev_states and current_state != "CA":
                analysis_parts.append(f"This person lived in California around the time you went to college together, then moved to {current_state}")
            elif "Los Angeles" in prev_cities or "LA" in prev_cities:
                analysis_parts.append(f"This person has Los Angeles in their address history, matching your UCLA connection")
        
        elif "high school" in relationship.lower() and "1993" in additional_context:
            if len(prev_states) > 0:
                analysis_parts.append(f"Geographic movement from {prev_states[0]} fits the timeline since high school graduation")
        
        # Marriage and name change detection
        if len(set(prev_states)) > 1:
            analysis_parts.append(f"Multiple state moves suggest major life changes like marriage or career")
    
    # Professional background story
    if result.employment_history:
        latest_job = result.employment_history[0]
        company = latest_job.get("company", "")
        if company:
            analysis_parts.append(f"Professional background at {company} indicates career progression since college")
    
    # Education verification
    if result.education_history:
        analysis_parts.append(f"Education history available for verification against your {relationship} connection")
    
    # Default if no specific patterns found
    if not analysis_parts:
        if relationship and first_name:
            return f"This {first_name} profile shows life patterns consistent with your {relationship} search"
        else:
            return "Basic information matches search criteria"
    
    return ". ".join(analysis_parts)

# =========================
# Mock Data for Testing
# =========================

def create_mock_enformion_results(params: PersonSearchParams) -> EnformionResponse:
    """
    Create mock results for testing when EnformionGo API is not available
    """
    mock_results = [
        EnformionResult(
            id="mock_1",
            confidence_score=0.87,
            first_name=params.first_name,
            last_name=params.last_name,
            age=32,
            current_address={"city": params.city or "Denver", "state": params.state or "CO"},
            previous_addresses=[
                {"city": "Austin", "state": "TX"},
                {"city": "College Station", "state": "TX"}
            ],
            relatives=[
                {"first_name": "John", "last_name": params.last_name},
                {"first_name": "Mary", "last_name": params.last_name}
            ],
            employment_history=[
                {"company": "Tech Corp", "title": "Marketing Manager", "years": "2020-Present"}
            ],
            education_history=[
                {"school": "Texas A&M University", "degree": "Bachelor's", "year": "2015"}
            ]
        ),
        EnformionResult(
            id="mock_2",
            confidence_score=0.64,
            first_name=params.first_name,
            last_name=params.last_name,
            age=35,
            current_address={"city": "Colorado Springs", "state": "CO"},
            previous_addresses=[
                {"city": "Phoenix", "state": "AZ"}
            ],
            relatives=[
                {"first_name": "Robert", "last_name": params.last_name}
            ],
            employment_history=[
                {"company": "School District", "title": "Teacher", "years": "2018-Present"}
            ]
        )
    ]
    
    return EnformionResponse(
        success=True,
        total_results=len(mock_results),
        results=mock_results,
        query_id="mock_query_123",
        credits_used=1
    )