#!/usr/bin/env python3
"""
Test script to debug Neo4j connection issues
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_env_vars():
    """Test if environment variables are properly set"""
    print("🔍 Checking environment variables...")
    
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    print(f"NEO4J_URI: {neo4j_uri}")
    print(f"NEO4J_PASSWORD: {'*' * len(neo4j_password) if neo4j_password else 'None'}")
    
    if not neo4j_uri:
        print("❌ NEO4J_URI is not set!")
        return False
    
    if not neo4j_password:
        print("❌ NEO4J_PASSWORD is not set!")
        return False
    
    print("✅ Environment variables are set")
    return True

def test_basic_connection():
    """Test basic Neo4j connection"""
    try:
        from neo4j import GraphDatabase
        
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        print("🔗 Testing basic Neo4j connection...")
        
        driver = GraphDatabase.driver(neo4j_uri, auth=("neo4j", neo4j_password))
        
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            test_value = result.single()["test"]
            
        driver.close()
        
        print(f"✅ Basic connection successful (test value: {test_value})")
        return True
        
    except Exception as e:
        print(f"❌ Basic connection failed: {e}")
        return False

def test_graph_store():
    """Test LlamaIndex Neo4j graph store connection"""
    try:
        print("📊 Testing LlamaIndex graph store...")
        
        from DB_neo4j import get_graph_store
        
        graph_store = get_graph_store()
        print("✅ Graph store connection successful")
        return True
        
    except Exception as e:
        print(f"❌ Graph store connection failed: {e}")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🧪 Neo4j Connection Test Suite")
    print("=" * 40)
    
    success = True
    
    # Test 1: Environment variables
    if not test_env_vars():
        success = False
        print("\n💡 Make sure your .env file contains:")
        print("NEO4J_URI=bolt://localhost:7687")  
        print("NEO4J_PASSWORD=your_password_here")
        return
    
    print()
    
    # Test 2: Basic connection
    if not test_basic_connection():
        success = False
        print("\n💡 Check if Neo4j is running and credentials are correct")
        return
    
    print()
    
    # Test 3: Graph store
    if not test_graph_store():
        success = False
    
    print()
    print("=" * 40)
    if success:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed. Check the output above.")

if __name__ == "__main__":
    main()
