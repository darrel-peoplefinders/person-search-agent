#!/usr/bin/env python3
"""
API Response Analysis Tool
Helps analyze long EnformionGo API responses and extract key insights
"""

import json
import sys
from datetime import datetime
from typing import Dict, List, Any

def analyze_enformion_response(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze EnformionGo API response and extract key insights"""
    
    analysis = {
        "summary": {},
        "people_found": [],
        "timeline_analysis": [],
        "data_quality": {},
        "recommendations": []
    }
    
    # Extract basic summary
    if "searchResults" in response_data:
        results = response_data["searchResults"]
        analysis["summary"] = {
            "total_results": len(results),
            "request_time": response_data.get("totalRequestExecutionTimeMs", 0),
            "request_id": response_data.get("requestId", "N/A")
        }
        
        # Analyze each person found
        for i, person in enumerate(results):
            person_analysis = analyze_person(person, i + 1)
            analysis["people_found"].append(person_analysis)
            
            # Add timeline analysis
            if person_analysis.get("timeline_issues"):
                analysis["timeline_analysis"].extend(person_analysis["timeline_issues"])
    
    # Data quality assessment
    analysis["data_quality"] = assess_data_quality(response_data)
    
    # Generate recommendations
    analysis["recommendations"] = generate_recommendations(analysis)
    
    return analysis

def analyze_person(person: Dict[str, Any], rank: int) -> Dict[str, Any]:
    """Analyze individual person record"""
    
    person_info = {
        "rank": rank,
        "name": person.get("fullName", "Unknown"),
        "age": person.get("age"),
        "locations": [],
        "contact_info": {
            "phones": len(person.get("phoneNumbers", [])),
            "emails": len(person.get("emailAddresses", [])),
            "addresses": len(person.get("addresses", []))
        },
        "timeline_issues": [],
        "confidence_factors": []
    }
    
    # Extract locations
    locations = person.get("locations", [])
    person_info["locations"] = [f"{loc.get('city', 'Unknown')}, {loc.get('state', 'Unknown')}" 
                               for loc in locations[:5]]  # Top 5 locations
    
    # Timeline analysis
    current_year = datetime.now().year
    age = person.get("age")
    
    if age:
        birth_year = current_year - age
        
        # Check for timeline inconsistencies
        addresses = person.get("addresses", [])
        for addr in addresses:
            first_reported = addr.get("firstReportedDate", "")
            if first_reported:
                try:
                    # Parse date format like "5/1/1993"
                    if "/" in first_reported:
                        parts = first_reported.split("/")
                        if len(parts) >= 3:
                            addr_year = int(parts[2])
                            age_at_address = addr_year - birth_year
                            
                            if age_at_address < 16:  # Too young for independent address
                                person_info["timeline_issues"].append({
                                    "issue": "Young age at address",
                                    "details": f"Age {age_at_address} at {addr.get('fullAddress', 'unknown address')} in {addr_year}",
                                    "severity": "medium"
                                })
                except (ValueError, IndexError):
                    pass
    
    # Confidence factors
    if person_info["contact_info"]["phones"] > 2:
        person_info["confidence_factors"].append("Multiple phone numbers")
    
    if person_info["contact_info"]["addresses"] > 5:
        person_info["confidence_factors"].append("Rich address history")
    
    if person.get("isPremium"):
        person_info["confidence_factors"].append("Premium record")
    
    return person_info

def assess_data_quality(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """Assess overall data quality of the response"""
    
    quality = {
        "overall_score": 0,
        "strengths": [],
        "weaknesses": [],
        "data_completeness": {}
    }
    
    results = response_data.get("searchResults", [])
    
    if not results:
        quality["overall_score"] = 0
        quality["weaknesses"].append("No results found")
        return quality
    
    # Calculate completeness scores
    total_people = len(results)
    people_with_phones = sum(1 for p in results if p.get("phoneNumbers"))
    people_with_emails = sum(1 for p in results if p.get("emailAddresses"))
    people_with_addresses = sum(1 for p in results if p.get("addresses"))
    people_with_age = sum(1 for p in results if p.get("age"))
    
    quality["data_completeness"] = {
        "phone_coverage": f"{(people_with_phones/total_people)*100:.1f}%",
        "email_coverage": f"{(people_with_emails/total_people)*100:.1f}%", 
        "address_coverage": f"{(people_with_addresses/total_people)*100:.1f}%",
        "age_coverage": f"{(people_with_age/total_people)*100:.1f}%"
    }
    
    # Determine strengths and weaknesses
    if people_with_phones / total_people > 0.8:
        quality["strengths"].append("High phone number coverage")
    else:
        quality["weaknesses"].append("Low phone number coverage")
    
    if people_with_addresses / total_people > 0.9:
        quality["strengths"].append("Excellent address coverage")
    
    if people_with_age / total_people > 0.7:
        quality["strengths"].append("Good age data coverage")
    
    # Calculate overall score (0-100)
    coverage_score = (people_with_phones + people_with_emails + people_with_addresses + people_with_age) / (total_people * 4) * 100
    quality["overall_score"] = min(100, coverage_score)
    
    return quality

def generate_recommendations(analysis: Dict[str, Any]) -> List[str]:
    """Generate actionable recommendations based on analysis"""
    
    recommendations = []
    
    # Based on timeline issues
    if analysis["timeline_analysis"]:
        recommendations.append("Implement timeline validation to flag suspicious age/address combinations")
        recommendations.append("Add confidence scoring that considers timeline consistency")
    
    # Based on data quality
    quality = analysis["data_quality"]
    if quality["overall_score"] < 70:
        recommendations.append("Consider supplementing with additional data sources")
    
    # Based on number of results
    result_count = analysis["summary"].get("total_results", 0)
    if result_count > 8:
        recommendations.append("Implement intelligent ranking to surface best matches first")
    elif result_count < 3:
        recommendations.append("Consider broader search criteria or fuzzy matching")
    
    # Performance recommendations
    request_time = analysis["summary"].get("request_time", 0)
    if request_time > 2000:  # 2 seconds
        recommendations.append("Optimize API performance - consider caching for common queries")
    
    return recommendations

def print_analysis(analysis: Dict[str, Any]):
    """Pretty print the analysis results"""
    
    print("=" * 60)
    print("ENFORMION API RESPONSE ANALYSIS")
    print("=" * 60)
    
    # Summary
    summary = analysis["summary"]
    print(f"\n📊 SUMMARY")
    print(f"   Results Found: {summary.get('total_results', 0)}")
    print(f"   Response Time: {summary.get('request_time', 0)}ms")
    print(f"   Request ID: {summary.get('request_id', 'N/A')}")
    
    # People found
    print(f"\n👥 PEOPLE FOUND ({len(analysis['people_found'])})")
    for person in analysis["people_found"][:5]:  # Top 5
        print(f"   #{person['rank']}: {person['name']}")
        print(f"      Age: {person.get('age', 'Unknown')}")
        print(f"      Locations: {', '.join(person['locations'][:3])}")
        print(f"      Contact: {person['contact_info']['phones']} phones, {person['contact_info']['emails']} emails")
        if person["confidence_factors"]:
            print(f"      Confidence: {', '.join(person['confidence_factors'])}")
        if person["timeline_issues"]:
            print(f"      ⚠️  Timeline Issues: {len(person['timeline_issues'])}")
        print()
    
    # Timeline analysis
    if analysis["timeline_analysis"]:
        print(f"\n⏰ TIMELINE ISSUES ({len(analysis['timeline_analysis'])})")
        for issue in analysis["timeline_analysis"][:5]:
            print(f"   ⚠️  {issue['issue']}: {issue['details']}")
    
    # Data quality
    quality = analysis["data_quality"]
    print(f"\n🎯 DATA QUALITY (Score: {quality['overall_score']:.1f}/100)")
    print(f"   Coverage:")
    for metric, value in quality["data_completeness"].items():
        print(f"      {metric.replace('_', ' ').title()}: {value}")
    
    if quality["strengths"]:
        print(f"   ✅ Strengths: {', '.join(quality['strengths'])}")
    if quality["weaknesses"]:
        print(f"   ❌ Weaknesses: {', '.join(quality['weaknesses'])}")
    
    # Recommendations
    if analysis["recommendations"]:
        print(f"\n💡 RECOMMENDATIONS")
        for i, rec in enumerate(analysis["recommendations"], 1):
            print(f"   {i}. {rec}")
    
    print("\n" + "=" * 60)

def main():
    """Main function to run analysis"""
    
    if len(sys.argv) < 2:
        print("Usage: python api_analyzer.py <json_file_or_paste_json>")
        print("\nTo analyze your API response:")
        print("1. Save the JSON response to a file")
        print("2. Run: python api_analyzer.py response.json")
        print("\nOr paste JSON directly when prompted...")
        
        print("\nPaste your JSON response here (press Enter twice when done):")
        lines = []
        while True:
            try:
                line = input()
                if line == "":
                    break
                lines.append(line)
            except EOFError:
                break
        
        json_text = "\n".join(lines)
    else:
        # Read from file
        try:
            with open(sys.argv[1], 'r') as f:
                json_text = f.read()
        except FileNotFoundError:
            # Maybe it's JSON directly passed
            json_text = sys.argv[1]
    
    try:
        # Parse JSON
        if json_text.strip().startswith('{'):
            response_data = json.loads(json_text)
        else:
            print("Error: Invalid JSON format")
            return
        
        # Analyze response
        analysis = analyze_enformion_response(response_data)
        
        # Print results
        print_analysis(analysis)
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
    except Exception as e:
        print(f"Error analyzing response: {e}")

if __name__ == "__main__":
    main()
