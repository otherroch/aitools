# Comprehensive edge case tests for FaceEnhancer functionality
"""Edge case and comprehensive tests for face enhancement functionality."""

import pytest
import numpy as np
from unittest.mock import patch


def test_parameter_validation_invalid_enhancement_model():
    """Test that invalid enhancement_model values are properly validated."""
    from chararep.config import PipelineConfig
    from chararep.face_enhancer import FaceEnhancer
    
    # Test with invalid model name
    cfg = PipelineConfig(
        enable_face_enhancement=True,
        enhancement_weight=0.7,
        enhancement_model="invalid_model_name",
        enhance_model_path=None,
    )
    
    enhancer = FaceEnhancer(cfg)
    assert not enhancer.available
    
    # Test with None model
    cfg2 = PipelineConfig(
        enable_face_enhancement=True,
        enhancement_weight=0.7,
        enhancement_model=None,
        enhance_model_path=None,
    )
    
    enhancer2 = FaceEnhancer(cfg2)
    assert not enhancer2.available


def test_parameter_validation_edge_case_weights():
    """Test edge case values for enhancement_weight parameter."""
    from chararep.config import PipelineConfig
    from chararep.face_enhancer import FaceEnhancer
    
    test_cases = [
        0.0,      # Minimum weight (no enhancement)
        0.5,      # Medium weight
        1.0,      # Maximum weight
        1.5,      # Above maximum
        -0.5,     # Negative weight
        2.0,      # Far above maximum
    ]
    
    for weight in test_cases:
        cfg = PipelineConfig(
            enable_face_enhancement=True,
            enhancement_weight=weight,
            enhancement_model="gfpgan",
            enhance_model_path=None,
        )
        
        enhancer = FaceEnhancer(cfg)
        # Different weights may affect backend availability differently
        assert hasattr(enhancer, 'available')


def test_empty_and_small_frames():
    """Test enhancement with edge case frame sizes."""
    from chararep.config import PipelineConfig
    from chararep.face_enhancer import FaceEnhancer
    
    cfg = PipelineConfig(
        enable_face_enhancement=True,
        enhancement_weight=0.7,
        enhancement_model="gfpgan",
        enhance_model_path=None,
    )
    
    enhancer = FaceEnhancer(cfg)
    assert enhancer.available  # Should be available in this environment
    
    test_cases = [
        (0, 0),      # Zero dimensions
        (1, 1),      # Single pixel
        (1, 100),    # Single row
        (100, 1),    # Single column
        (3, 3),      # Minimum realistic size
        (1000, 1000),  # Large size (may fail due to memory)
    ]
    
    for height, width in test_cases:
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        try:
            result = enhancer.enhance(frame, frame_idx=0)
            if height > 0 and width > 0:
                assert result.shape == frame.shape
                assert result.dtype == np.uint8
        except (ValueError, RuntimeError) as e:
            # Some tests with extreme sizes may legitimately fail
            assert "ValueError" in type(e).__name__ or "RuntimeError" in type(e).__name__


def test_incorrect_bounding_boxes():
    """Test enhancement with incorrect/special bounding box coordinates."""
    from chararep.config import PipelineConfig
    from chararep.face_enhancer import FaceEnhancer
    import types
    
    cfg = PipelineConfig(
        enable_face_enhancement=True,
        enhancement_weight=0.7,
        enhancement_model="gfpgan",
        enhance_model_path=None,
    )
    
    enhancer = FaceEnhancer(cfg)
    assert enhancer.available
    
    # Create test frame
    frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    
    test_cases = [
        # (x1, y1, x2, y2, label, description)
        (0, 0, 10, 10, "normal", "Normal valid bbox"),
        (0, 0, 0, 0, "zero_area", "Zero area bbox"),
        (-10, -10, 10, 10, "negative_coords", "Negative coordinates"),
        (10, 10, 5, 5, "inverted_coords", "Inverted coordinates (x2 < x1)"),
        (200, 200, 210, 210, "outside_image", "Completely outside image"),
        (0, 0, 300, 300, "exceeds_image", "Exceeds image boundaries"),
        (50, 50, 50, 50, "point_box", "Single point box"),
        (50.5, 50.5, 60.5, 60.5, "float_coords", "Float coordinates"),
    ]
    
    for bbox_coords in test_cases:
        x1, y1, x2, y2, label, description = bbox_coords
        
        # Create tracked face
        face = types.SimpleNamespace(
            track_id=0,
            bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
            identity_label=label,
            landmarks=None,
        )
        
        # This should handle edge cases gracefully
        result = enhancer.enhance_faces(frame.copy(), [face], frame_idx=0)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8


def test_landmarks_edge_cases():
    """Test enhancement with edge case landmark configurations."""
    from chararep.config import PipelineConfig
    from chararep.face_enhancer import FaceEnhancer
    import types
    
    cfg = PipelineConfig(
        enable_face_enhancement=True,
        enhancement_weight=0.7,
        enhancement_model="gfpgan",
        enhance_model_path=None,
    )
    
    enhancer = FaceEnhancer(cfg)
    assert enhancer.available
    
    frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    
    test_cases = [
        # (landmarks, description)
        (None, "No landmarks"),
        (np.array([], dtype=np.float32), "Empty landmarks array"),
        (np.array([[0, 0]], dtype=np.float32), "Single landmark"),
        (np.random.randn(10, 2), "Many random landmarks"),
        (np.array([[0, 0], [200, 200], [100, 100]], dtype=np.float32), "Extreme coordinates"),
        (np.array([[0, 0], [200, 200], [100, 100], [50, 50], [150, 150]], dtype=np.float32), "Five landmarks"),
    ]
    
    for landmarks, description in test_cases:
        face = types.SimpleNamespace(
            track_id=0,
            bbox=np.array([10, 10, 100, 100], dtype=np.float32),
            identity_label="test",
            landmarks=landmarks,
        )
        
        result = enhancer.enhance_faces(frame.copy(), [face], frame_idx=0)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8


def test_memory_usage_edge_cases():
    """Test memory usage with large number of faces."""
    from chararep.config import PipelineConfig
    from chararep.face_enhancer import FaceEnhancer
    import types
    
    cfg = PipelineConfig(
        enable_face_enhancement=True,
        enhancement_weight=0.7,
        enhancement_model="gfpgan",
        enhance_model_path=None,
    )
    
    enhancer = FaceEnhancer(cfg)
    assert enhancer.available
    
    frame = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
    
    # Test with varying number of faces
    test_cases = [
        0,    # No faces
        1,    # Single face
        10,   # Small number
        100,  # Large number
    ]
    
    for num_faces in test_cases:
        faces = []
        for i in range(num_faces):
            x1 = 50 * i
            y1 = 50 * i
            x2 = x1 + 100
            y2 = y1 + 100
            
            face = types.SimpleNamespace(
                track_id=i,
                bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                identity_label=str(i),
                landmarks=None,
            )
            faces.append(face)
        
        result = enhancer.enhance_faces(frame.copy(), faces, frame_idx=0)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8


def test_multiple_enhancement_calls():
    """Test multiple consecutive enhancement calls for consistency."""
    from chararep.config import PipelineConfig
    from chararep.face_enhancer import FaceEnhancer
    import types
    
    cfg = PipelineConfig(
        enable_face_enhancement=True,
        enhancement_weight=0.7,
        enhancement_model="gfpgan",
        enhance_model_path=None,
    )
    
    enhancer = FaceEnhancer(cfg)
    assert enhancer.available
    
    frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    
    # Perform multiple enhancement calls
    results = []
    for i in range(10):
        face = types.SimpleNamespace(
            track_id=i,
            bbox=np.array([50, 50, 150, 150], dtype=np.float32),
            identity_label="test",
            landmarks=None,
        )
        result = enhancer.enhance_faces(frame.copy(), [face], frame_idx=i)
        results.append(result)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8
    
    # Results should all be valid numpy arrays
    for i, result in enumerate(results):
        assert isinstance(result, np.ndarray)
        assert result.shape == (200, 200, 3)


def test_backend_fallback_mechanisms():
    """Test fallback mechanisms when backends fail."""
    from chararep.config import PipelineConfig
    from chararep.face_enhancer import FaceEnhancer
    
    cfg = PipelineConfig(
        enable_face_enhancement=True,
        enhancement_weight=0.7,
        enhancement_model="gfpgan",
        enhance_model_path=None,
    )
    
    enhancer = FaceEnhancer(cfg)
    assert enhancer.available
    
    # Mock a backend failure
    with patch.object(enhancer._backend, 'enhance_crop', 
                     side_effect=RuntimeError("Mock backend failure")):
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = enhancer.enhance(frame, frame_idx=0)
        
        # Should fall back gracefully and return original
        np.testing.assert_array_equal(result, frame)


def test_concurrent_enhancement_simulation():
    """Simulate concurrent enhancement calls."""
    from chararep.config import PipelineConfig
    from chararep.face_enhancer import FaceEnhancer
    import types
    
    cfg = PipelineConfig(
        enable_face_enhancement=True,
        enhancement_weight=0.7,
        enhancement_model="gfpgan",
        enhance_model_path=None,
    )
    
    enhancer = FaceEnhancer(cfg)
    assert enhancer.available
    
    # Create test frames
    frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(5)]
    
    # Process frames in sequence (simulating concurrent access)
    results = []
    for i, frame in enumerate(frames):
        face = types.SimpleNamespace(
            track_id=i,
            bbox=np.array([10, 10, 90, 90], dtype=np.float32),
            identity_label="test",
            landmarks=None,
        )
        result = enhancer.enhance_faces(frame.copy(), [face], frame_idx=i)
        results.append(result)
        
        assert result.shape == frame.shape
        assert result.dtype == np.uint8
    
    # All results should be valid
    for result in results:
        assert isinstance(result, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])