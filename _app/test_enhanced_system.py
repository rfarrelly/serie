def test_enhanced_system():
    """Test the enhanced system with sample data"""
    try:
        from main import create_enhanced_application

        print("Testing enhanced application creation...")
        app = create_enhanced_application()
        print("✓ Enhanced application created successfully")

        print("Testing prediction generation...")
        predictions, opportunities = app.app_service.generate_enhanced_predictions()
        print(
            f"✓ Generated {len(predictions)} predictions and {len(opportunities)} opportunities"
        )

        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


# Run the test
if __name__ == "__main__":
    test_enhanced_system()
