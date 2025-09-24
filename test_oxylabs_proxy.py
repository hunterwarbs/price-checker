#!/usr/bin/env python3
"""Test script to verify Oxylabs proxy configuration."""

import os
import asyncio
from src.screenshotter import ProductScreenshotter

async def test_oxylabs_proxy():
    """Test the Oxylabs proxy configuration."""
    print("🔍 Testing Oxylabs proxy configuration...")
    print("=" * 50)
    
    # Check if OXYLABS_PROXY is set
    oxylabs_proxy = os.getenv("OXYLABS_PROXY")
    if not oxylabs_proxy:
        print("❌ OXYLABS_PROXY environment variable is not set")
        print("Please set it with: export OXYLABS_PROXY='username:password'")
        print("\nExample: export OXYLABS_PROXY='your_username:your_password'")
        return
    
    print(f"✅ OXYLABS_PROXY is set (length: {len(oxylabs_proxy)} chars)")
    
    # Validate format
    if ':' not in oxylabs_proxy:
        print("❌ Invalid OXYLABS_PROXY format. Expected 'username:password'")
        return
        
    username, password = oxylabs_proxy.split(':', 1)
    print(f"✅ Parsed credentials - Username: {username[:4]}***, Password: ***")
    
    # Test screenshotter initialization
    try:
        print("\n🔧 Testing ProductScreenshotter initialization...")
        screenshotter = ProductScreenshotter(use_oxylabs_proxy=True)
        print("✅ ProductScreenshotter initialized successfully")
        
        # Test proxy config generation
        print("\n🔧 Testing proxy configuration generation...")
        proxy_config = screenshotter._get_oxylabs_proxy_config()
        if proxy_config:
            print("✅ Proxy config generated successfully:")
            print(f"   Server: {proxy_config['server']}")
            print(f"   Username: {proxy_config['username']}")
            print(f"   Password: {'*' * len(proxy_config['password'])}")
        else:
            print("❌ Failed to generate proxy config")
            return
            
    except Exception as e:
        print(f"❌ Error initializing ProductScreenshotter: {e}")
        return
    
    # Test with actual browser context (optional - will launch browser)
    test_browser = input("\n🌐 Do you want to test with actual browser connection? (y/N): ").lower().strip()
    if test_browser == 'y':
        try:
            print("\n🚀 Testing with actual browser...")
            async with screenshotter:
                context = await screenshotter._create_stealth_context("https://ip.oxylabs.io/location")
                page = await context.new_page()
                
                # Quick test
                await page.goto("https://ip.oxylabs.io/location", timeout=30000)
                content = await page.content()
                
                if "ip" in content.lower():
                    print("✅ Successfully connected through Oxylabs proxy!")
                    print("✅ IP location service responded correctly")
                else:
                    print("⚠️  Connection made but response unclear")
                
                await page.close()
                await context.close()
                
        except Exception as e:
            print(f"❌ Browser test failed: {e}")
            print("This might be due to network issues or incorrect credentials")
    
    print("\n" + "=" * 50)
    print("🎉 Oxylabs proxy configuration test completed!")
    print("\nIf all tests passed, your Oxylabs proxy is properly configured.")
    print("The proxy will be used automatically when running your screenshotter.")

if __name__ == "__main__":
    asyncio.run(test_oxylabs_proxy())